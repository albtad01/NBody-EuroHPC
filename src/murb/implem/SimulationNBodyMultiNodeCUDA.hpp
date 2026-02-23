#ifndef SIMULATION_N_BODY_MULTINODE_CUDA_HPP_
#define SIMULATION_N_BODY_MULTINODE_CUDA_HPP_

#ifdef USE_CUDA

#include "core/SimulationNBodyInterface.hpp"
#include "core/CUDABodies.hpp"
#include <mpi.h>
#include <vector>
#include <memory>

template <typename T>
class SimulationNBodyMultiNodeCUDA : public SimulationNBodyInterface<T> {
protected:
    int rank, size;
    std::vector<int> counts;
    std::vector<int> displs;

    std::shared_ptr<CUDABodies<T>> cudaBodiesPtr;

    // Buffer Device
    devAccSoA_t<T> devAccelerations;
    T* devGM;
    T softSquared; // Variabile aggiunta per evitare l'errore di compilazione

    // Send Buffers per isolare la memoria ed evitare Buffer Aliasing (CUDA Error 700)
    T *send_qx, *send_qy, *send_qz;
    T *send_vx, *send_vy, *send_vz;

    int _num_threads;
    int _elem_per_thread;
    int _num_blocks;

    MPI_Datatype mpi_type();
    void initMPI_and_GPU();
    void buildCountsDispls(int n);
    void syncStateMPI();

public:
    SimulationNBodyMultiNodeCUDA(const BodiesAllocatorInterface<T>& allocator, const T soft);
    virtual ~SimulationNBodyMultiNodeCUDA();

    virtual void computeOneIteration() override;
};

#endif // USE_CUDA
#endif // SIMULATION_N_BODY_MULTINODE_CUDA_HPP_