// #ifndef SIMULATION_N_BODY_MULTI_NODE_HPP_
// #define SIMULATION_N_BODY_MULTI_NODE_HPP_

// #include <vector>
// #include <cstddef>
// #include <mpi.h>

// #include "core/SimulationNBodyInterface.hpp"

// // MPI + OpenMP implementation (multinode)
// // Tag: mpi+omp (o "multinode" come decidi tu nel factory)
// template <typename T>
// class SimulationNBodyMultiNode : public SimulationNBodyInterface<T>
// {
// protected:
//     std::vector<accAoS_t<T>> accelerations;

//     // buffers globali (replicati su tutti i rank)
//     std::vector<T> qx_all, qy_all, qz_all, m_all;

//     // displs/counts per Allgatherv
//     std::vector<int> counts, displs;

//     int rank = 0;
//     int size = 1;

// public:
//     SimulationNBodyMultiNode(const BodiesAllocatorInterface<T>& allocator, const T soft = (T)0.035);
//     ~SimulationNBodyMultiNode() override = default;

//     void computeOneIteration() override;
//     const std::vector<accAoS_t<T>>& getAccAoS() { return accelerations; }

// protected:
//     void initMPI();
//     void buildCountsDispls(int n);
//     void gatherGlobalState();
//     void gatherGlobalAccelerations(int i0, int i1);

//     void computeBodiesAcceleration_mpi();

//     // helper: map C++ type to MPI datatype
//     static MPI_Datatype mpi_type();
// };

// #endif
