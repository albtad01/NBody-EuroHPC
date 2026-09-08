#include "MultiGpuRuntime.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace murb {
namespace {

void checkCudaRuntime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("gpu+multinode ") + operation + ": " +
                                 cudaGetErrorString(result));
}

} // namespace

void checkMpi(int result, const char* operation) {
    if (result == MPI_SUCCESS) return;
    std::array<char, MPI_MAX_ERROR_STRING> message{};
    int length = 0;
    MPI_Error_string(result, message.data(), &length);
    throw std::runtime_error(std::string("gpu+multinode ") + operation + ": " +
                             std::string(message.data(), static_cast<std::size_t>(length)));
}

MultiGpuRuntime::MultiGpuRuntime(
    int& argc, char**& argv, bool requireFourGpuNode, bool diagnostics) try {
    int initialized = 0;
    if (MPI_Initialized(&initialized) != MPI_SUCCESS)
        throw std::runtime_error("gpu+multinode MPI_Initialized failed");
    if (!initialized) {
        const int result = MPI_Init(&argc, &argv);
        if (result != MPI_SUCCESS)
            throw std::runtime_error("gpu+multinode MPI_Init failed with code " +
                                     std::to_string(result));
        ownsMpi = true;
    }

    checkMpi(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN),
             "MPI_Comm_set_errhandler");
    checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &runtimeInfo.worldRank), "MPI_Comm_rank");
    checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &runtimeInfo.worldSize), "MPI_Comm_size");
    checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                                 runtimeInfo.worldRank, MPI_INFO_NULL, &localComm),
             "MPI_Comm_split_type");
    checkMpi(MPI_Comm_set_errhandler(localComm, MPI_ERRORS_RETURN),
             "local MPI_Comm_set_errhandler");
    checkMpi(MPI_Comm_rank(localComm, &runtimeInfo.localRank), "local MPI_Comm_rank");
    checkMpi(MPI_Comm_size(localComm, &runtimeInfo.localSize), "local MPI_Comm_size");

    if (requireFourGpuNode &&
        (runtimeInfo.worldSize != 4 || runtimeInfo.localSize != 4)) {
        if (runtimeInfo.worldRank == 0)
            std::cerr << "gpu+multinode requires four MPI ranks on one node; world_size="
                      << runtimeInfo.worldSize << " local_size=" << runtimeInfo.localSize
                      << std::endl;
        abort(EXIT_FAILURE);
    }

    int processorLength = 0;
    std::array<char, MPI_MAX_PROCESSOR_NAME> processor{};
    checkMpi(MPI_Get_processor_name(processor.data(), &processorLength),
             "MPI_Get_processor_name");
    runtimeInfo.processorName.assign(processor.data(),
                                     static_cast<std::size_t>(processorLength));

    checkCudaRuntime(cudaGetDeviceCount(&runtimeInfo.visibleDeviceCount),
                     "cudaGetDeviceCount");
    if (runtimeInfo.visibleDeviceCount != 1)
        throw std::runtime_error(
            "gpu+multinode requires exactly one Slurm-visible GPU per MPI rank; rank " +
            std::to_string(runtimeInfo.worldRank) + " sees " +
            std::to_string(runtimeInfo.visibleDeviceCount));

    runtimeInfo.cudaDevice = 0;
    checkCudaRuntime(cudaSetDevice(runtimeInfo.cudaDevice), "cudaSetDevice(0)");
    cudaDeviceProp properties{};
    checkCudaRuntime(cudaGetDeviceProperties(&properties, runtimeInfo.cudaDevice),
                     "cudaGetDeviceProperties");
    runtimeInfo.deviceName = properties.name;
    std::array<char, 32> busId{};
    checkCudaRuntime(cudaDeviceGetPCIBusId(busId.data(), static_cast<int>(busId.size()),
                                           runtimeInfo.cudaDevice),
                     "cudaDeviceGetPCIBusId");
    runtimeInfo.pciBusId = busId.data();
    if (const char* visible = std::getenv("CUDA_VISIBLE_DEVICES"))
        runtimeInfo.visibleDevices = visible;
    else
        runtimeInfo.visibleDevices = "unset";

    std::vector<char> allBusIds(static_cast<std::size_t>(runtimeInfo.localSize) * busId.size());
    checkMpi(MPI_Allgather(busId.data(), static_cast<int>(busId.size()), MPI_CHAR,
                           allBusIds.data(), static_cast<int>(busId.size()), MPI_CHAR,
                           localComm),
             "MPI_Allgather(PCI bus IDs)");
    for (int left = 0; left < runtimeInfo.localSize; ++left) {
        for (int right = left + 1; right < runtimeInfo.localSize; ++right) {
            const char* leftId = allBusIds.data() + static_cast<std::size_t>(left) * busId.size();
            const char* rightId = allBusIds.data() + static_cast<std::size_t>(right) * busId.size();
            if (std::strncmp(leftId, rightId, busId.size()) == 0)
                throw std::runtime_error("gpu+multinode MPI ranks selected the same physical GPU");
        }
    }

    if (diagnostics) {
        for (int outputRank = 0; outputRank < runtimeInfo.worldSize; ++outputRank) {
            checkMpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(mapping)");
            if (runtimeInfo.worldRank == outputRank) {
                std::cout << "MPI rank " << runtimeInfo.worldRank
                          << " local_rank=" << runtimeInfo.localRank
                          << " node=" << runtimeInfo.processorName
                          << " CUDA_VISIBLE_DEVICES=" << runtimeInfo.visibleDevices
                          << " cuda_device=" << runtimeInfo.cudaDevice
                          << " gpu=" << runtimeInfo.deviceName
                          << " pci=" << runtimeInfo.pciBusId << std::endl;
            }
        }
        checkMpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(mapping completion)");
    }
} catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    int initialized = 0;
    int alreadyFinalized = 0;
    MPI_Initialized(&initialized);
    if (initialized) MPI_Finalized(&alreadyFinalized);
    if (initialized && !alreadyFinalized) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    throw;
}

MultiGpuRuntime::~MultiGpuRuntime() {
    if (finalized) return;
    if (localComm != MPI_COMM_NULL) MPI_Comm_free(&localComm);
    if (ownsMpi) MPI_Finalize();
}

const MultiGpuRuntimeInfo& MultiGpuRuntime::info() const { return runtimeInfo; }

void MultiGpuRuntime::finalize() {
    if (finalized) return;
    if (localComm != MPI_COMM_NULL)
        checkMpi(MPI_Comm_free(&localComm), "MPI_Comm_free");
    if (ownsMpi) checkMpi(MPI_Finalize(), "MPI_Finalize");
    finalized = true;
}

[[noreturn]] void MultiGpuRuntime::abort(int errorCode) noexcept {
    MPI_Abort(MPI_COMM_WORLD, errorCode);
    std::abort();
}

} // namespace murb
