// #include "SimulationNBodyMultiNode.hpp"

// #include <cmath>
// #include <algorithm>
// #include <type_traits>

// #include "mipp.h"

// #ifdef _OPENMP
// #include <omp.h>
// #endif

// #ifndef MURB_MPI_RSQRT_NR
// #define MURB_MPI_RSQRT_NR 0
// #endif

// template <typename T>
// static inline mipp::Reg<T> rsqrt_fast(const mipp::Reg<T>& x)
// {
//     mipp::Reg<T> y = mipp::rsqrt(x);
//     if constexpr (std::is_same<T, float>::value) {
// #if MURB_MPI_RSQRT_NR >= 1
//         const mipp::Reg<T> half = T(0.5f);
//         const mipp::Reg<T> threehalfs = T(1.5f);
//         y = y * (threehalfs - half * x * y * y);
// #endif
// #if MURB_MPI_RSQRT_NR >= 2
//         const mipp::Reg<T> half = T(0.5f);
//         const mipp::Reg<T> threehalfs = T(1.5f);
//         y = y * (threehalfs - half * x * y * y);
// #endif
//     }
//     return y;
// }

// template <typename T>
// MPI_Datatype SimulationNBodyMultiNode<T>::mpi_type()
// {
//     if constexpr (std::is_same<T, float>::value)  return MPI_FLOAT;
//     if constexpr (std::is_same<T, double>::value) return MPI_DOUBLE;
//     return MPI_BYTE; // fallback (non dovrebbe mai accadere)
// }

// template <typename T>
// SimulationNBodyMultiNode<T>::SimulationNBodyMultiNode(const BodiesAllocatorInterface<T>& allocator, const T soft)
//     : SimulationNBodyInterface<T>(allocator, soft)
// {
//     this->flopsPerIte = 20.f * (T)this->getBodies()->getN() * (T)this->getBodies()->getN();
//     accelerations.resize(this->getBodies()->getN());

//     initMPI();
//     buildCountsDispls((int)this->getBodies()->getN());

//     const int n = (int)this->getBodies()->getN();
//     qx_all.resize(n);
//     qy_all.resize(n);
//     qz_all.resize(n);
//     m_all.resize(n);
// }

// template <typename T>
// void SimulationNBodyMultiNode<T>::initMPI()
// {
//     int inited = 0;
//     MPI_Initialized(&inited);
//     if (!inited) {
//         // se il framework non inizializza MPI, lo facciamo noi
//         int argc = 0;
//         char** argv = nullptr;
//         MPI_Init(&argc, &argv);
//     }
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);
// }

// template <typename T>
// void SimulationNBodyMultiNode<T>::buildCountsDispls(int n)
// {
//     counts.assign(size, 0);
//     displs.assign(size, 0);

//     const int base = n / size;
//     const int rem  = n % size;

//     for (int r = 0; r < size; ++r) {
//         counts[r] = base + (r < rem ? 1 : 0);
//     }
//     displs[0] = 0;
//     for (int r = 1; r < size; ++r) {
//         displs[r] = displs[r-1] + counts[r-1];
//     }
// }

// template <typename T>
// void SimulationNBodyMultiNode<T>::gatherGlobalState()
// {
//     const auto& d = this->getBodies()->getDataSoA();
//     const T* qx = d.qx.data();
//     const T* qy = d.qy.data();
//     const T* qz = d.qz.data();
//     const T* m  = d.m.data();

//     const int n = (int)this->getBodies()->getN();

//     // ognuno contribuisce con il suo “chunk” [displs[rank], displs[rank]+counts[rank])
//     MPI_Allgatherv(qx + displs[rank], counts[rank], mpi_type(),
//                    qx_all.data(), counts.data(), displs.data(), mpi_type(), MPI_COMM_WORLD);

//     MPI_Allgatherv(qy + displs[rank], counts[rank], mpi_type(),
//                    qy_all.data(), counts.data(), displs.data(), mpi_type(), MPI_COMM_WORLD);

//     MPI_Allgatherv(qz + displs[rank], counts[rank], mpi_type(),
//                    qz_all.data(), counts.data(), displs.data(), mpi_type(), MPI_COMM_WORLD);

//     MPI_Allgatherv(m + displs[rank], counts[rank], mpi_type(),
//                    m_all.data(), counts.data(), displs.data(), mpi_type(), MPI_COMM_WORLD);

//     (void)n;
// }

// template <typename T>
// void SimulationNBodyMultiNode<T>::gatherGlobalAccelerations(int i0, int i1)
// {
//     // raccogliamo accelerazioni locali in accelerations globali replicati
//     // packaging: usiamo 3 allgatherv su array contigui per semplicità
//     const int n = (int)this->getBodies()->getN();

//     std::vector<T> ax_loc(i1 - i0), ay_loc(i1 - i0), az_loc(i1 - i0);
//     for (int ii = i0; ii < i1; ++ii) {
//         const int k = ii - i0;
//         ax_loc[k] = accelerations[ii].ax;
//         ay_loc[k] = accelerations[ii].ay;
//         az_loc[k] = accelerations[ii].az;
//     }

//     std::vector<T> ax_all(n), ay_all(n), az_all(n);

//     MPI_Allgatherv(ax_loc.data(), counts[rank], mpi_type(),
//                    ax_all.data(), counts.data(), displs.data(), mpi_type(), MPI_COMM_WORLD);

//     MPI_Allgatherv(ay_loc.data(), counts[rank], mpi_type(),
//                    ay_all.data(), counts.data(), displs.data(), mpi_type(), MPI_COMM_WORLD);

//     MPI_Allgatherv(az_loc.data(), counts[rank], mpi_type(),
//                    az_all.data(), counts.data(), displs.data(), mpi_type(), MPI_COMM_WORLD);

//     for (int i = 0; i < n; ++i) {
//         accelerations[i].ax = ax_all[i];
//         accelerations[i].ay = ay_all[i];
//         accelerations[i].az = az_all[i];
//     }
// }

// template <typename T>
// void SimulationNBodyMultiNode<T>::computeBodiesAcceleration_mpi()
// {
//     const int n = (int)this->getBodies()->getN();
//     const int i0 = displs[rank];
//     const int i1 = i0 + counts[rank];

//     // 1) global state (posizioni + masse) replicato
//     gatherGlobalState();

//     const T soft2 = this->soft * this->soft;
//     const T G = this->G;

//     const int V = mipp::N<T>();
//     const int n_vec = n - (n % V);

// #ifdef _OPENMP
//     omp_set_dynamic(0);
// #pragma omp parallel for schedule(static)
// #endif
//     for (int i = i0; i < i1; ++i) {

//         const T qi_x = qx_all[i];
//         const T qi_y = qy_all[i];
//         const T qi_z = qz_all[i];

//         const mipp::Reg<T> r_qi_x = qi_x;
//         const mipp::Reg<T> r_qi_y = qi_y;
//         const mipp::Reg<T> r_qi_z = qi_z;
//         const mipp::Reg<T> r_soft = soft2;
//         const mipp::Reg<T> r_G    = G;

//         mipp::Reg<T> r_ax = T(0), r_ay = T(0), r_az = T(0);

//         int j = 0;
//         for (; j < n_vec; j += V) {
//             const mipp::Reg<T> r_mj  = &m_all[j];
//             const mipp::Reg<T> r_qjx = &qx_all[j];
//             const mipp::Reg<T> r_qjy = &qy_all[j];
//             const mipp::Reg<T> r_qjz = &qz_all[j];

//             const mipp::Reg<T> r_dx = r_qjx - r_qi_x;
//             const mipp::Reg<T> r_dy = r_qjy - r_qi_y;
//             const mipp::Reg<T> r_dz = r_qjz - r_qi_z;

//             const mipp::Reg<T> r_dist2 =
//                 mipp::fmadd(r_dx, r_dx,
//                 mipp::fmadd(r_dy, r_dy,
//                 mipp::fmadd(r_dz, r_dz, r_soft)));

//             const mipp::Reg<T> r_inv  = rsqrt_fast<T>(r_dist2);
//             const mipp::Reg<T> r_inv2 = r_inv * r_inv;
//             const mipp::Reg<T> r_inv3 = r_inv2 * r_inv;

//             const mipp::Reg<T> r_fac = r_G * r_mj * r_inv3;

//             r_ax = mipp::fmadd(r_fac, r_dx, r_ax);
//             r_ay = mipp::fmadd(r_fac, r_dy, r_ay);
//             r_az = mipp::fmadd(r_fac, r_dz, r_az);
//         }

//         T ax = mipp::hadd(r_ax);
//         T ay = mipp::hadd(r_ay);
//         T az = mipp::hadd(r_az);

//         for (; j < n; ++j) {
//             const T dx = qx_all[j] - qi_x;
//             const T dy = qy_all[j] - qi_y;
//             const T dz = qz_all[j] - qi_z;

//             const T dist2 = dx*dx + dy*dy + dz*dz + soft2;
//             const T inv = (T)1 / std::sqrt(dist2);
//             const T inv3 = (inv * inv) * inv;
//             const T fac = G * m_all[j] * inv3;

//             ax += fac * dx;
//             ay += fac * dy;
//             az += fac * dz;
//         }

//         accelerations[i].ax = ax;
//         accelerations[i].ay = ay;
//         accelerations[i].az = az;
//     }

//     // 2) ricostruisci accelerazioni globali
//     gatherGlobalAccelerations(i0, i1);
// }

// template <typename T>
// void SimulationNBodyMultiNode<T>::computeOneIteration()
// {
//     computeBodiesAcceleration_mpi();
//     // tutti i rank hanno accelerazioni globali identiche → update deterministico ovunque
//     this->bodies->updatePositionsAndVelocities(accelerations, this->dt);
// }

// // explicit instantiation
// template class SimulationNBodyMultiNode<float>;
// template class SimulationNBodyMultiNode<double>;
