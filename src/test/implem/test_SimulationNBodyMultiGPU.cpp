#include <catch.hpp>

#if defined(USE_CUDA) && defined(USE_MPI)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>

#include <mpi.h>

#include "MultiGpuRuntime.hpp"
#include "SimulationNBodyMultiNodeCUDA.hpp"
#include "SimulationNBodyNaive.hpp"

namespace {

float relativeDifference(float reference, float candidate) {
    const float scale = std::max(1.0f, std::max(std::abs(reference), std::abs(candidate)));
    return std::abs(reference - candidate) / scale;
}

void compareMultiGpuWithNaive(std::size_t bodyCount,
                              const std::string& scheme,
                              std::size_t iterations,
                              float tolerance) {
    constexpr float Softening = 2e+08f;
    constexpr float Timestep = 3600.0f;

    BodiesAllocator<float> referenceAllocator(bodyCount, scheme);
    SimulationNBodyNaive<float> reference(referenceAllocator, Softening);
    reference.setDt(Timestep);

    CUDABodiesAllocator<float> candidateAllocator(bodyCount, scheme);
    SimulationNBodyMultiNodeCUDA<float> candidate(candidateAllocator, Softening);
    candidate.setDt(Timestep);
    int rank = 0;
    murb::checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank(correctness)");

    for (std::size_t iteration = 0; iteration <= iterations; ++iteration) {
        if (iteration > 0) {
            reference.computeOneIteration();
            candidate.computeOneIteration();
        }

        const auto& expected = reference.getBodies()->getDataSoA();
        const auto& actual = candidate.getBodies()->getDataSoA();
        float localMaximum = 0.0f;
        for (std::size_t body = 0; body < bodyCount; ++body) {
            const std::array<float, 6> differences = {
                relativeDifference(expected.qx[body], actual.qx[body]),
                relativeDifference(expected.qy[body], actual.qy[body]),
                relativeDifference(expected.qz[body], actual.qz[body]),
                relativeDifference(expected.vx[body], actual.vx[body]),
                relativeDifference(expected.vy[body], actual.vy[body]),
                relativeDifference(expected.vz[body], actual.vz[body])};
            localMaximum = std::max(localMaximum,
                                    *std::max_element(differences.begin(), differences.end()));
        }

        float globalMaximum = 0.0f;
        murb::checkMpi(MPI_Allreduce(&localMaximum, &globalMaximum, 1, MPI_FLOAT, MPI_MAX,
                                     MPI_COMM_WORLD),
                       "MPI_Allreduce(correctness error)");
        if (rank == 0)
            std::cout << "gpu+multinode correctness scheme=" << scheme
                      << " bodies=" << bodyCount << " iteration=" << iteration
                      << " max_normalized_error=" << globalMaximum
                      << " tolerance=" << (iteration == 0 ? 0.0f : tolerance) << '\n';
        CAPTURE(scheme, bodyCount, iteration, tolerance, globalMaximum);
        REQUIRE(globalMaximum <= (iteration == 0 ? 0.0f : tolerance));
    }
}

} // namespace

TEST_CASE("gpu+multinode matches cpu+naive on four local GPUs",
          "[correctness][gpu-multinode]") {
    int worldSize = 0;
    int localSize = 0;
    MPI_Comm localComm = MPI_COMM_NULL;
    murb::checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &worldSize), "MPI_Comm_size(test)");
    murb::checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                       MPI_INFO_NULL, &localComm),
                   "MPI_Comm_split_type(test)");
    murb::checkMpi(MPI_Comm_size(localComm, &localSize), "local MPI_Comm_size(test)");
    murb::checkMpi(MPI_Comm_free(&localComm), "MPI_Comm_free(test)");
    if (worldSize != 4 || localSize != 4) {
        WARN("gpu+multinode correctness requires four MPI ranks on one node");
        return;
    }

    compareMultiGpuWithNaive(2048, "random", 1, 1e-3f);
    compareMultiGpuWithNaive(2049, "random", 3, 1e-3f);
    compareMultiGpuWithNaive(2051, "galaxy", 3, 1e-1f);
}

#endif
