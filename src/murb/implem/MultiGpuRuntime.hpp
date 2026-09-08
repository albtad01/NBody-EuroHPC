#ifndef MULTI_GPU_RUNTIME_HPP_
#define MULTI_GPU_RUNTIME_HPP_

#if !defined(USE_CUDA) || !defined(USE_MPI)
#error "MultiGpuRuntime requires CUDA and MPI support"
#endif

#include <mpi.h>

#include <string>

namespace murb {

struct MultiGpuRuntimeInfo {
    int worldRank = 0;
    int worldSize = 1;
    int localRank = 0;
    int localSize = 1;
    int cudaDevice = 0;
    int visibleDeviceCount = 0;
    std::string processorName;
    std::string deviceName;
    std::string pciBusId;
    std::string visibleDevices;
};

void checkMpi(int result, const char* operation);

class MultiGpuRuntime {
  public:
    MultiGpuRuntime(int& argc, char**& argv, bool requireFourGpuNode, bool diagnostics);
    ~MultiGpuRuntime();

    MultiGpuRuntime(const MultiGpuRuntime&) = delete;
    MultiGpuRuntime& operator=(const MultiGpuRuntime&) = delete;

    const MultiGpuRuntimeInfo& info() const;
    void finalize();
    [[noreturn]] void abort(int errorCode) noexcept;

  private:
    MPI_Comm localComm = MPI_COMM_NULL;
    MultiGpuRuntimeInfo runtimeInfo;
    bool ownsMpi = false;
    bool finalized = false;
};

} // namespace murb

#endif
