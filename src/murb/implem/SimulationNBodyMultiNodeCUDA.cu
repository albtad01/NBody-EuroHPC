#include "SimulationNBodyMultiNodeCUDA.hpp"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

// =========================================================================
// MACRO E HELPER DEVICE
// =========================================================================
#define CUDA_CHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char *file, int line) {
    if (error_code != cudaSuccess) {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n",
                (int)error_code, cudaGetErrorString(error_code), file, line);
        exit((int)error_code);
    }
}

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
// LOGICA C++ HOST (MultiNode CUDA-Aware)
// =========================================================================

template <typename T>
MPI_Datatype SimulationNBodyMultiNodeCUDA<T>::mpi_type() {
    if constexpr (std::is_same<T, float>::value)  return MPI_FLOAT;
    if constexpr (std::is_same<T, double>::value) return MPI_DOUBLE;
    return MPI_BYTE;
}

template <typename T>
void SimulationNBodyMultiNodeCUDA<T>::initMPI_and_GPU() {
    int inited = 0;
    MPI_Initialized(&inited);
    if (!inited) {
        int argc = 0; char** argv = nullptr;
        MPI_Init(&argc, &argv);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    CUDA_CHECK(cudaSetDevice(rank % num_gpus));
}

template <typename T>
void SimulationNBodyMultiNodeCUDA<T>::buildCountsDispls(int n) {
    counts.assign(size, 0);
    displs.assign(size, 0);
    const int base = n / size;
    const int rem  = n % size;
    for (int r = 0; r < size; ++r) { counts[r] = base + (r < rem ? 1 : 0); }
    displs[0] = 0;
    for (int r = 1; r < size; ++r) { displs[r] = displs[r-1] + counts[r-1]; }
}

template <typename T>
SimulationNBodyMultiNodeCUDA<T>::SimulationNBodyMultiNodeCUDA(const BodiesAllocatorInterface<T>& allocator, const T soft)
    : SimulationNBodyInterface<T>(allocator, soft), softSquared(soft*soft)
{
    initMPI_and_GPU();
    
    int n = (int)this->getBodies()->getN();
    this->flopsPerIte = 20.f * (T)n * (T)n;
    
    // 1. Calcoliamo counts PRIMA di usare local_n
    buildCountsDispls(n);
    int local_n = counts[rank];

    CUDA_CHECK(cudaMalloc(&this->devAccelerations.x, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&this->devAccelerations.y, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&this->devAccelerations.z, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&this->devGM, n * sizeof(T)));

    // 2. Alloca i Send Buffer (Ora local_n ha il valore corretto!)
    if (local_n > 0) {
        CUDA_CHECK(cudaMalloc(&this->send_qx, local_n * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&this->send_qy, local_n * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&this->send_qz, local_n * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&this->send_vx, local_n * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&this->send_vy, local_n * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&this->send_vz, local_n * sizeof(T)));
    } else {
        this->send_qx = this->send_qy = this->send_qz = nullptr;
        this->send_vx = this->send_vy = this->send_vz = nullptr;
    }

    this->cudaBodiesPtr = std::dynamic_pointer_cast<CUDABodies<T>>(this->bodies);

    int init_blocks = (n + 255) / 256;
    devInitializeDevGM_MPI<T><<<init_blocks, 256>>>(this->cudaBodiesPtr->getDevDataSoA(), n, this->G, this->devGM);
    CUDA_CHECK(cudaDeviceSynchronize());

    this->_num_threads = 256;
    this->_elem_per_thread = 4;
    
    if (local_n > 0)
        this->_num_blocks = (local_n + (this->_num_threads * this->_elem_per_thread) - 1) / (this->_num_threads * this->_elem_per_thread);
    else 
        this->_num_blocks = 0;
}

template <typename T>
void SimulationNBodyMultiNodeCUDA<T>::syncStateMPI() {
    auto devData = this->cudaBodiesPtr->getDevDataSoA();
    int offset = displs[rank];
    int count = counts[rank];

    if (count == 0) return;

    // 1. Copia sicura nei Send Buffers
    CUDA_CHECK(cudaMemcpy(send_qx, devData.qx + offset, count * sizeof(T), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(send_qy, devData.qy + offset, count * sizeof(T), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(send_qz, devData.qz + offset, count * sizeof(T), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(send_vx, devData.vx + offset, count * sizeof(T), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(send_vy, devData.vy + offset, count * sizeof(T), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(send_vz, devData.vz + offset, count * sizeof(T), cudaMemcpyDeviceToDevice));
    
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Datatype type = mpi_type();
    
    // 2. AllGather
    MPI_Allgatherv(send_qx, count, type, devData.qx, counts.data(), displs.data(), type, MPI_COMM_WORLD);
    MPI_Allgatherv(send_qy, count, type, devData.qy, counts.data(), displs.data(), type, MPI_COMM_WORLD);
    MPI_Allgatherv(send_qz, count, type, devData.qz, counts.data(), displs.data(), type, MPI_COMM_WORLD);
    MPI_Allgatherv(send_vx, count, type, devData.vx, counts.data(), displs.data(), type, MPI_COMM_WORLD);
    MPI_Allgatherv(send_vy, count, type, devData.vy, counts.data(), displs.data(), type, MPI_COMM_WORLD);
    MPI_Allgatherv(send_vz, count, type, devData.vz, counts.data(), displs.data(), type, MPI_COMM_WORLD);
}

template <typename T>
void SimulationNBodyMultiNodeCUDA<T>::computeOneIteration() {
    int total_n = (int)this->bodies->getN();
    int local_n = counts[rank];
    int offset = displs[rank];

    if (local_n > 0) {
        devComputeBodiesAccelerationMPI<T><<<this->_num_blocks, this->_num_threads>>>(
            this->devAccelerations, this->cudaBodiesPtr->getDevDataSoA(), this->devGM,
            total_n, offset, local_n, this->softSquared
        );
        CUDA_CHECK(cudaGetLastError());

        int update_blocks = (local_n + 255) / 256;
        devUpdateKinematicsMPI<T><<<update_blocks, 256>>>(
            this->cudaBodiesPtr->getDevDataSoA(), this->devAccelerations, this->dt, offset, local_n
        );
        CUDA_CHECK(cudaGetLastError());
    }

    syncStateMPI();
}

template <typename T>
SimulationNBodyMultiNodeCUDA<T>::~SimulationNBodyMultiNodeCUDA() {
    cudaFree(devAccelerations.x); cudaFree(devAccelerations.y); cudaFree(devAccelerations.z);
    cudaFree(devGM);
    
    if (send_qx) {
        cudaFree(send_qx); cudaFree(send_qy); cudaFree(send_qz);
        cudaFree(send_vx); cudaFree(send_vy); cudaFree(send_vz);
    }
}

template class SimulationNBodyMultiNodeCUDA<float>;
template class SimulationNBodyMultiNodeCUDA<double>;

#endif // USE_CUDA