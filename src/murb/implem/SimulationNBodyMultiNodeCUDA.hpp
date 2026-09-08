#ifndef SIMULATION_N_BODY_MULTINODE_CUDA_HPP_
#define SIMULATION_N_BODY_MULTINODE_CUDA_HPP_

#if defined(USE_CUDA) && defined(USE_MPI)

#include <mpi.h>

#include <memory>
#include <vector>

#include "core/CUDABodies.hpp"
#include "core/SimulationNBodyInterface.hpp"

template <typename T>
class SimulationNBodyMultiNodeCUDA : public SimulationNBodyInterface<T> {
  public:
    SimulationNBodyMultiNodeCUDA(const BodiesAllocatorInterface<T>& allocator, T soft);
    ~SimulationNBodyMultiNodeCUDA() override;

    void computeOneIteration() override;

  private:
    static MPI_Datatype mpiType();
    void buildCountsAndDisplacements(int bodyCount);
    void synchronizeGlobalState();

    int rank = 0;
    int size = 1;
    std::vector<int> counts;
    std::vector<int> displacements;

    std::shared_ptr<CUDABodies<T>> cudaBodies;
    devAccSoA_t<T> deviceAccelerations{};
    T* deviceGM = nullptr;
    T softSquared;

    std::vector<T> localQx, localQy, localQz;
    std::vector<T> localVx, localVy, localVz;
    std::vector<T> globalQx, globalQy, globalQz;
    std::vector<T> globalVx, globalVy, globalVz;

    int threadsPerBlock = 256;
    int elementsPerThread = 4;
    int blockCount = 0;
};

#endif
#endif
