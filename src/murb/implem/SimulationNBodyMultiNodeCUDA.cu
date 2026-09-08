#include "SimulationNBodyMultiNodeCUDA.hpp"

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "MultiGpuRuntime.hpp"

// =========================================================================
// MACRO E HELPER DEVICE
// =========================================================================
namespace {
void checkMultiCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("gpu+multinode ") + operation + ": " +
                                 cudaGetErrorString(result));
}
}

#define CUDA_CHECK(call) checkMultiCuda((call), #call)

template <typename T> __device__ __forceinline__ T device_rsqrt(T val);
template <> __device__ __forceinline__ float device_rsqrt<float>(float val) { return rsqrtf(val); }
template <> __device__ __forceinline__ double device_rsqrt<double>(double val) { return rsqrt(val); }

template <typename T> __device__ __forceinline__ T fmadd(T a, T b, T c) { return a*b + c; }
template <> __device__ __forceinline__ float fmadd<float>(float a, float b, float c) { return fmaf(a,b,c); }
template <> __device__ __forceinline__ double fmadd<double>(double a, double b, double c) { return fma(a,b,c); }

// =========================================================================
// KERNEL: Inizializzazione G * M
// =========================================================================
template <typename T>
__global__ void devInitializeDevGM_MPI(const devDataSoA_t<T> devDataSoA, const int n_bodies, T G, T* __restrict__ devGM) {
    const int iBody = blockIdx.x * blockDim.x + threadIdx.x;
    if (iBody < n_bodies) devGM[iBody] = G * devDataSoA.m[iBody];
}

// =========================================================================
// KERNEL: Computazione Gravità
// =========================================================================
template <typename T>
__global__ __launch_bounds__(256, 2)
void devComputeBodiesAccelerationMPI(
    devAccSoA_t<T> devAcc,
    const devDataSoA_t<T> devData,
    const T* __restrict__ devGM,
    const int total_bodies,
    const int offset,
    const int local_n,
    const T softSquared
) {
    constexpr int BLOCK = 256;
    constexpr int EPT = 4;
    constexpr int TILE = BLOCK * EPT;

    __shared__ T SHm[TILE];
    __shared__ T SHqx[TILE];
    __shared__ T SHqy[TILE];
    __shared__ T SHqz[TILE];

    T accX[EPT], accY[EPT], accZ[EPT];
    T rix[EPT],  riy[EPT],  riz[EPT];

    #pragma unroll
    for (int k = 0; k < EPT; ++k) {
        accX[k] = accY[k] = accZ[k] = T(0);
        int local_idx = blockIdx.x * (BLOCK * EPT) + threadIdx.x + k * BLOCK;
        int global_idx = offset + local_idx;
        
        if (local_idx < local_n) {
            rix[k] = devData.qx[global_idx];
            riy[k] = devData.qy[global_idx];
            riz[k] = devData.qz[global_idx];
        } else {
            rix[k] = riy[k] = riz[k] = T(0);
        }
    }

    for (int base = 0; base < total_bodies; base += TILE) {
        for (int t = threadIdx.x; t < TILE; t += BLOCK) {
            int j = base + t;
            if (j < total_bodies) {
                SHm[t]  = devGM[j];
                SHqx[t] = devData.qx[j];
                SHqy[t] = devData.qy[j];
                SHqz[t] = devData.qz[j];
            } else {
                SHm[t] = SHqx[t] = SHqy[t] = SHqz[t] = T(0);
            }
        }
        __syncthreads();

        #pragma unroll 4
        for (int j = 0; j < TILE; ++j) {
            const T sjx = SHqx[j]; const T sjy = SHqy[j]; const T sjz = SHqz[j]; const T sm = SHm[j];
            #pragma unroll
            for (int k = 0; k < EPT; ++k) {
                const T rijx = sjx - rix[k];
                const T rijy = sjy - riy[k];
                const T rijz = sjz - riz[k];

                T distSq = fmadd(rijx, rijx, softSquared);
                distSq   = fmadd(rijy, rijy, distSq);
                distSq   = fmadd(rijz, rijz, distSq);

                const T invDist  = device_rsqrt<T>(distSq);
                const T f = sm * (invDist * invDist * invDist);

                accX[k] = fmadd(f, rijx, accX[k]);
                accY[k] = fmadd(f, rijy, accY[k]);
                accZ[k] = fmadd(f, rijz, accZ[k]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int k = 0; k < EPT; ++k) {
        int local_idx = blockIdx.x * (BLOCK * EPT) + threadIdx.x + k * BLOCK;
        int global_idx = offset + local_idx;
        if (local_idx < local_n) {
            devAcc.x[global_idx] = accX[k];
            devAcc.y[global_idx] = accY[k];
            devAcc.z[global_idx] = accZ[k];
        }
    }
}

// =========================================================================
// KERNEL: Aggiornamento Cinematica
// =========================================================================
template <typename T>
__global__ void devUpdateKinematicsMPI(devDataSoA_t<T> devData, const devAccSoA_t<T> devAcc, T dt, int offset, int local_n) {
    int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_idx >= local_n) return;
    
    int i = offset + local_idx;

    T vx = devData.vx[i]; T vy = devData.vy[i]; T vz = devData.vz[i];
    T axDt = devAcc.x[i] * dt; T ayDt = devAcc.y[i] * dt; T azDt = devAcc.z[i] * dt;

    devData.qx[i] += (vx + axDt * 0.5f) * dt;
    devData.qy[i] += (vy + ayDt * 0.5f) * dt;
    devData.qz[i] += (vz + azDt * 0.5f) * dt;

    devData.vx[i] = vx + axDt;
    devData.vy[i] = vy + ayDt;
    devData.vz[i] = vz + azDt;
}

// =========================================================================
// HOST ORCHESTRATION (explicit host-staged MPI)
// =========================================================================

template <typename T>
MPI_Datatype SimulationNBodyMultiNodeCUDA<T>::mpiType() {
    if constexpr (std::is_same_v<T, float>) return MPI_FLOAT;
    else if constexpr (std::is_same_v<T, double>) return MPI_DOUBLE;
    else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "gpu+multinode supports only float and double");
        return MPI_BYTE;
    }
}

template <typename T>
void SimulationNBodyMultiNodeCUDA<T>::buildCountsAndDisplacements(int bodyCount) {
    counts.resize(static_cast<std::size_t>(size));
    displacements.resize(static_cast<std::size_t>(size));
    const int quotient = bodyCount / size;
    const int remainder = bodyCount % size;
    int displacement = 0;
    for (int partition = 0; partition < size; ++partition) {
        counts[static_cast<std::size_t>(partition)] =
            quotient + (partition < remainder ? 1 : 0);
        displacements[static_cast<std::size_t>(partition)] = displacement;
        displacement += counts[static_cast<std::size_t>(partition)];
    }
    if (displacement != bodyCount)
        throw std::logic_error("gpu+multinode partition does not cover every body");
}

template <typename T>
SimulationNBodyMultiNodeCUDA<T>::SimulationNBodyMultiNodeCUDA(
    const BodiesAllocatorInterface<T>& allocator, T soft)
    : SimulationNBodyInterface<T>(allocator, soft), softSquared(soft * soft) {
    int initialized = 0;
    checkMpi(MPI_Initialized(&initialized), "MPI_Initialized");
    if (!initialized)
        throw std::logic_error("gpu+multinode requires MPI_Init before construction");
    checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &size), "MPI_Comm_size");

    const auto bodyCount = this->getBodies()->getN();
    if (bodyCount == 0 ||
        bodyCount > static_cast<unsigned long>(std::numeric_limits<int>::max() - 1024))
        throw std::invalid_argument("gpu+multinode body count exceeds supported dimensions");
    const int totalBodies = static_cast<int>(bodyCount);
    buildCountsAndDisplacements(totalBodies);

    cudaBodies = std::dynamic_pointer_cast<CUDABodies<T>>(this->bodies);
    if (!cudaBodies)
        throw std::invalid_argument("gpu+multinode requires CUDABodiesAllocator");

    const int localBodies = counts[static_cast<std::size_t>(rank)];
    localQx.resize(static_cast<std::size_t>(localBodies));
    localQy.resize(static_cast<std::size_t>(localBodies));
    localQz.resize(static_cast<std::size_t>(localBodies));
    localVx.resize(static_cast<std::size_t>(localBodies));
    localVy.resize(static_cast<std::size_t>(localBodies));
    localVz.resize(static_cast<std::size_t>(localBodies));
    globalQx.resize(bodyCount);
    globalQy.resize(bodyCount);
    globalQz.resize(bodyCount);
    globalVx.resize(bodyCount);
    globalVy.resize(bodyCount);
    globalVz.resize(bodyCount);

    CUDA_CHECK(cudaMalloc(&deviceAccelerations.x, bodyCount * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&deviceAccelerations.y, bodyCount * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&deviceAccelerations.z, bodyCount * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&deviceGM, bodyCount * sizeof(T)));

    const int initializationBlocks = (totalBodies + threadsPerBlock - 1) / threadsPerBlock;
    devInitializeDevGM_MPI<T><<<initializationBlocks, threadsPerBlock>>>(
        cudaBodies->getDevDataSoA(), totalBodies, this->G, deviceGM);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    blockCount = (localBodies + threadsPerBlock * elementsPerThread - 1) /
                 (threadsPerBlock * elementsPerThread);
    this->flopsPerIte = T{20} * static_cast<T>(bodyCount) * static_cast<T>(bodyCount);
}

template <typename T>
void SimulationNBodyMultiNodeCUDA<T>::synchronizeGlobalState() {
    const auto& deviceData = cudaBodies->getDevDataSoA();
    const int localBodies = counts[static_cast<std::size_t>(rank)];
    const int localOffset = displacements[static_cast<std::size_t>(rank)];
    const auto localBytes = static_cast<std::size_t>(localBodies) * sizeof(T);
    const auto globalBytes = static_cast<std::size_t>(this->getBodies()->getN()) * sizeof(T);

    if (localBodies > 0) {
        CUDA_CHECK(cudaMemcpy(localQx.data(), deviceData.qx + localOffset,
                              localBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(localQy.data(), deviceData.qy + localOffset,
                              localBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(localQz.data(), deviceData.qz + localOffset,
                              localBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(localVx.data(), deviceData.vx + localOffset,
                              localBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(localVy.data(), deviceData.vy + localOffset,
                              localBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(localVz.data(), deviceData.vz + localOffset,
                              localBytes, cudaMemcpyDeviceToHost));
    }

    const MPI_Datatype type = mpiType();
    checkMpi(MPI_Allgatherv(localQx.data(), localBodies, type,
                            globalQx.data(), counts.data(), displacements.data(), type,
                            MPI_COMM_WORLD),
             "MPI_Allgatherv(qx host staging)");
    checkMpi(MPI_Allgatherv(localQy.data(), localBodies, type,
                            globalQy.data(), counts.data(), displacements.data(), type,
                            MPI_COMM_WORLD),
             "MPI_Allgatherv(qy host staging)");
    checkMpi(MPI_Allgatherv(localQz.data(), localBodies, type,
                            globalQz.data(), counts.data(), displacements.data(), type,
                            MPI_COMM_WORLD),
             "MPI_Allgatherv(qz host staging)");
    checkMpi(MPI_Allgatherv(localVx.data(), localBodies, type,
                            globalVx.data(), counts.data(), displacements.data(), type,
                            MPI_COMM_WORLD),
             "MPI_Allgatherv(vx host staging)");
    checkMpi(MPI_Allgatherv(localVy.data(), localBodies, type,
                            globalVy.data(), counts.data(), displacements.data(), type,
                            MPI_COMM_WORLD),
             "MPI_Allgatherv(vy host staging)");
    checkMpi(MPI_Allgatherv(localVz.data(), localBodies, type,
                            globalVz.data(), counts.data(), displacements.data(), type,
                            MPI_COMM_WORLD),
             "MPI_Allgatherv(vz host staging)");

    CUDA_CHECK(cudaMemcpy(deviceData.qx, globalQx.data(), globalBytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceData.qy, globalQy.data(), globalBytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceData.qz, globalQz.data(), globalBytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceData.vx, globalVx.data(), globalBytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceData.vy, globalVy.data(), globalBytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceData.vz, globalVz.data(), globalBytes,
                          cudaMemcpyHostToDevice));
    cudaBodies->invalidateDataSoA();
}

template <typename T>
void SimulationNBodyMultiNodeCUDA<T>::computeOneIteration() {
    const int totalBodies = static_cast<int>(this->getBodies()->getN());
    const int localBodies = counts[static_cast<std::size_t>(rank)];
    const int localOffset = displacements[static_cast<std::size_t>(rank)];

    if (localBodies > 0) {
        devComputeBodiesAccelerationMPI<T><<<blockCount, threadsPerBlock>>>(
            deviceAccelerations, cudaBodies->getDevDataSoA(), deviceGM,
            totalBodies, localOffset, localBodies, softSquared);
        CUDA_CHECK(cudaGetLastError());

        const int updateBlocks = (localBodies + threadsPerBlock - 1) / threadsPerBlock;
        devUpdateKinematicsMPI<T><<<updateBlocks, threadsPerBlock>>>(
            cudaBodies->getDevDataSoA(), deviceAccelerations, this->dt,
            localOffset, localBodies);
        CUDA_CHECK(cudaGetLastError());
    }

    synchronizeGlobalState();
}

template <typename T>
SimulationNBodyMultiNodeCUDA<T>::~SimulationNBodyMultiNodeCUDA() {
    cudaFree(deviceAccelerations.x);
    cudaFree(deviceAccelerations.y);
    cudaFree(deviceAccelerations.z);
    cudaFree(deviceGM);
}

template class SimulationNBodyMultiNodeCUDA<float>;
template class SimulationNBodyMultiNodeCUDA<double>;

#endif // USE_CUDA
