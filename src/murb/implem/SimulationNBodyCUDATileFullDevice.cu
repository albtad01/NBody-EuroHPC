#ifndef SIMULATION_N_BODY_CUDA_TILE_FULL_DEVICE_CU_
#define SIMULATION_N_BODY_CUDA_TILE_FULL_DEVICE_CU_

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>

#include <cuda_runtime.h>

#include "SimulationNBodyCUDATileFullDevice.hpp"

#define CUDA_CHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

// =================================== TEMPLATE of SQRT ====================================
template <typename T>
__device__ __forceinline__ T device_rsqrt(T val);

template <>
__device__ __forceinline__ float device_rsqrt<float>(float val) { return rsqrtf(val); }

template <>
__device__ __forceinline__ double device_rsqrt<double>(double val) { return rsqrt(val); }

// ============================= CONSTRUCTOR & its UTILITIES ================================

template <typename T> 
__global__ void devInitializeDevGM(const devDataSoA_t<T> devDataSoA, const int n_bodies, T G, T* devGM, const int elem_per_thread) {
    for (int k = 0; k < elem_per_thread; k++) {
        const int iBody = blockIdx.x * blockDim.x * elem_per_thread + threadIdx.x + k * blockDim.x;
        if ( iBody < n_bodies ) {
            devGM[iBody] = G * devDataSoA.m[iBody];
        }
    }
}

template <typename T>
SimulationNBodyCUDATileFullDevice<T>::SimulationNBodyCUDATileFullDevice(
        const BodiesAllocatorInterface<T>& allocator, const T soft, const bool transfer_each_iteration)
    : SimulationNBodyInterface<T>(allocator, soft), softSquared{soft*soft}, 
      transfer_each_iteration{transfer_each_iteration}
{
    this->flopsPerIte = 20.f * (T)this->getBodies()->getN() * (T)this->getBodies()->getN();

    const int NUM_SM = 128;
    int n = (int)this->getBodies()->getN();
    // if ( n <= 128 * 256 ) {
    //     this->_num_threads = 256;
    // } else if ( n <= 128 * 512 ) {
    //     this->_num_threads = 512;
    // } else {
    //     this->_num_threads = 1024;
    // }

    this->_num_threads = 1024;

    this->_num_blocks = std::min(NUM_SM, (n + this->_num_threads - 1) / this->_num_threads);
    int total_threads = this->_num_blocks * this->_num_threads;
    this->_elem_per_thread = (n + total_threads - 1) / total_threads;

    CUDA_CHECK(cudaMalloc(&this->devAccelerations.ax, this->getBodies()->getN()*sizeof(T)));
    CUDA_CHECK(cudaMalloc(&this->devAccelerations.ay, this->getBodies()->getN()*sizeof(T)));
    CUDA_CHECK(cudaMalloc(&this->devAccelerations.az, this->getBodies()->getN()*sizeof(T)));
    CUDA_CHECK(cudaMalloc(&this->devGM, this->getBodies()->getN()*sizeof(T)));

    // std::cout << "Number of threads: " << _num_threads << std::endl;
    // std::cout << "Number of blocks: " << _num_blocks << std::endl;
    // std::cout << "Element per thread: " << _elem_per_thread << std::endl;
    // std::cout << "Total desired bodies: " << total_threads * this->_elem_per_thread << std::endl;

    this->cudaBodiesPtr = std::dynamic_pointer_cast<CUDABodies<T>>(this->bodies);

    if ( this->cudaBodiesPtr == nullptr ) {
        std::cout << "Error in converting to CUDABodies!!!" << std::endl;
    }

    // cudaDeviceProp prop;
    // CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    // size_t l2_size = prop.l2CacheSize;

    // CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_size));

    devInitializeDevGM<T><<<this->_num_blocks,this->_num_threads>>>(this->cudaBodiesPtr->getDevDataSoA(), 
        this->bodies->getN(),this->G,this->devGM,this->_elem_per_thread);
}

// template <typename T>
// __global__ void devInitIterationTileFullDevice(devAccSoA_t<T> devAccelerations,
//                                  const int n_bodies) {
//     int iBody = blockIdx.x * blockDim.x + threadIdx.x;
//     if ( iBody < n_bodies ) {
//         devAccelerations.ax[iBody] = T(0);
//         devAccelerations.ay[iBody] = T(0);
//         devAccelerations.az[iBody] = T(0);
//     }
// }

template <typename T>
void SimulationNBodyCUDATileFullDevice<T>::initIteration()
{
    // int n = (int)this->getBodies()->getN();
    // int threads = 256; 
    // int blocks = (n + threads - 1) / threads;  
    
    // devInitIterationTileFullDevice<T><<<blocks, threads>>>(devAccelerations, n);
    
    // CUDA_CHECK(cudaGetLastError());    
}

template <typename T>
void SimulationNBodyCUDATileFullDevice<T>::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    this->cudaBodiesPtr->updatePositionsAndVelocitiesOnDevice(this->devAccelerations, this->dt);
    if ( this->transfer_each_iteration ) {
        this->cudaBodiesPtr->getDataSoA();
    }
}

// Old version before switching to multiple bodies per each thread
// WARNING: it works the best on -n 200k
template <typename T>
__global__ void devComputeBodiesAccelerationTileFullDevice200k(
    devAccSoA_t<T> devAccelerations, 
    const devDataSoA_t<T> devDataSoA,
    T* devGM,
    const int n_bodies,
    const T G,
    const T softSquared
) {
    constexpr int TILE_SIZE = 1024;

    __shared__ T SHm[TILE_SIZE];
    __shared__ T SHqx[TILE_SIZE];
    __shared__ T SHqy[TILE_SIZE];
    __shared__ T SHqz[TILE_SIZE];

    int iBody = blockIdx.x * blockDim.x + threadIdx.x;

    T accX = T(0), accY = T(0), accZ = T(0);

    T rix = T(0), riy = T(0), riz = T(0);
    if (iBody < n_bodies) {
        rix = devDataSoA.qx[iBody];
        riy = devDataSoA.qy[iBody];
        riz = devDataSoA.qz[iBody];
    }

    for (int base_idx = 0; base_idx < n_bodies; base_idx += TILE_SIZE) {
        for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
            int gg_idx = base_idx + i;
            if (gg_idx < n_bodies) {
                SHm[i]  = devGM[gg_idx];
                SHqx[i] = devDataSoA.qx[gg_idx];
                SHqy[i] = devDataSoA.qy[gg_idx];
                SHqz[i] = devDataSoA.qz[gg_idx];
            } else {
                SHm[i]  = T(0);
                SHqx[i] = T(0);
                SHqy[i] = T(0);
                SHqz[i] = T(0);
            }
        }

        __syncthreads();

        if (iBody < n_bodies) {
            
            #pragma unroll 32
            for (int jBody = 0; jBody < TILE_SIZE; jBody++) {
                const T rijx = SHqx[jBody] - rix;
                const T rijy = SHqy[jBody] - riy;
                const T rijz = SHqz[jBody] - riz;

                const T rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;

                const T inv = device_rsqrt<T>(rijSquared + softSquared);
                const T factor = inv * inv * inv;
                
                const T ai = factor * SHm[jBody];
                accX += ai * rijx;
                accY += ai * rijy;
                accZ += ai * rijz;
            }
        }

        __syncthreads();
    }

    if (iBody < n_bodies) {
        devAccelerations.ax[iBody] = accX;
        devAccelerations.ay[iBody] = accY;
        devAccelerations.az[iBody] = accZ;
    }
}

template <typename T>
__global__ void devComputeBodiesAccelerationTileFullDevice(
    devAccSoA_t<T> devAccelerations, 
    const devDataSoA_t<T> devDataSoA,
    const T* devGM,
    const int n_bodies,
    const T G,
    const T softSquared,
    const int elem_per_thread
){
    constexpr int N = 1;
    constexpr int TILE_SIZE = N * 1024;

    __shared__ T SHm[TILE_SIZE];
    __shared__ T SHqx[TILE_SIZE];
    __shared__ T SHqy[TILE_SIZE];
    __shared__ T SHqz[TILE_SIZE];

    for (int k = 0; k < elem_per_thread; k++) {
        const int iBody = blockIdx.x * blockDim.x * elem_per_thread + threadIdx.x + k * blockDim.x;
        
        T accX = T(0), accY = T(0), accZ = T(0);
        T rix = T(0), riy = T(0), riz = T(0);
        
        if (iBody < n_bodies) {
            rix = devDataSoA.qx[iBody];
            riy = devDataSoA.qy[iBody];
            riz = devDataSoA.qz[iBody];
        }

        for (int base_idx = 0; base_idx < n_bodies; base_idx += TILE_SIZE) {

            for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
                int gg_idx = base_idx + i;
                if (gg_idx < n_bodies) {
                    SHm[i]  = devGM[gg_idx];
                    SHqx[i] = devDataSoA.qx[gg_idx];
                    SHqy[i] = devDataSoA.qy[gg_idx];
                    SHqz[i] = devDataSoA.qz[gg_idx];
                } else {
                    SHm[i]  = T(0);
                    SHqx[i] = T(0);
                    SHqy[i] = T(0);
                    SHqz[i] = T(0);
                }
            }
            __syncthreads();

            if (iBody < n_bodies) {
                #pragma unroll 32
                for (int jBody = 0; jBody < TILE_SIZE; jBody++) {
                    const T rijx = SHqx[jBody] - rix;
                    const T rijy = SHqy[jBody] - riy;
                    const T rijz = SHqz[jBody] - riz;

                    const T rijSquared = rijx*rijx + rijy*rijy + rijz*rijz;

                    const T inv = device_rsqrt<T>(rijSquared + softSquared);
                    const T ai = inv * inv * inv * SHm[jBody];

                    accX += ai * rijx;
                    accY += ai * rijy;
                    accZ += ai * rijz;

                    #ifdef COMPUTE_ENERGY
                    
                    #endif
                }
            }
            __syncthreads();
        }

        if (iBody < n_bodies) {
            devAccelerations.ax[iBody] = accX;
            devAccelerations.ay[iBody] = accY;
            devAccelerations.az[iBody] = accZ;
        }
    }
}

// NOT PERFORMING GOOD
template <typename T>
__global__ void devComputeBodiesAccelerationTileFullDevice_2elem(
    devAccSoA_t<T> devAccelerations, 
    const devDataSoA_t<T> devDataSoA,
    const int n_bodies,
    const T G,
    const T softSquared,
    const int elem_per_thread
){
    constexpr int N = 1;
    constexpr int TILE_SIZE = N * 1024;

    __shared__ T SHm[TILE_SIZE];
    __shared__ T SHqx[TILE_SIZE];
    __shared__ T SHqy[TILE_SIZE];
    __shared__ T SHqz[TILE_SIZE];

    for (int k = 1; k <= elem_per_thread; k *= 2) {
        const int iBody1 = blockIdx.x * blockDim.x * 2 + (k-1) * threadIdx.x;
        const int iBody2 = blockIdx.x * blockDim.x * 2 + (k-1) * threadIdx.x + blockDim.x;
        
        T accX1 = T(0), accY1 = T(0), accZ1 = T(0);
        T accX2 = T(0), accY2 = T(0), accZ2 = T(0);
        T rix1 = T(0), riy1 = T(0), riz1 = T(0);
        T rix2 = T(0), riy2 = T(0), riz2 = T(0);
        
        if (iBody1 < n_bodies) {
            rix1 = devDataSoA.qx[iBody1];
            riy1 = devDataSoA.qy[iBody1];
            riz1 = devDataSoA.qz[iBody1];
        }
        
        if (iBody2 < n_bodies) {
            rix2 = devDataSoA.qx[iBody2];
            riy2 = devDataSoA.qy[iBody2];
            riz2 = devDataSoA.qz[iBody2];
        }

        for (int base_idx = 0; base_idx < n_bodies; base_idx += TILE_SIZE) {

            for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
                int gg_idx = base_idx + i;
                if (gg_idx < n_bodies) {
                    SHm[i]  = devDataSoA.m[gg_idx];
                    SHqx[i] = devDataSoA.qx[gg_idx];
                    SHqy[i] = devDataSoA.qy[gg_idx];
                    SHqz[i] = devDataSoA.qz[gg_idx];
                } else {
                    SHm[i]  = T(0);
                    SHqx[i] = T(0);
                    SHqy[i] = T(0);
                    SHqz[i] = T(0);
                }
            }
            __syncthreads();

                #pragma unroll 32
                for (int jBody = 0; jBody < TILE_SIZE; jBody++) {
                    // First element
                    const T rijx1 = SHqx[jBody] - rix1;
                    const T rijy1 = SHqy[jBody] - riy1;
                    const T rijz1 = SHqz[jBody] - riz1;

                    const T rijSquared1 = rijx1*rijx1 + rijy1*rijy1 + rijz1*rijz1;

                    const T inv1 = device_rsqrt<T>(rijSquared1 + softSquared);
                    const T factor1 = inv1 * inv1 * inv1;

                    const T ai1 = factor1 * SHm[jBody];
                    accX1 += ai1 * rijx1;
                    accY1 += ai1 * rijy1;
                    accZ1 += ai1 * rijz1;

                    // Second element
                    const T rijx2 = SHqx[jBody] - rix2;
                    const T rijy2 = SHqy[jBody] - riy2;
                    const T rijz2 = SHqz[jBody] - riz2;

                    const T rijSquared2 = rijx2*rijx2 + rijy2*rijy2 + rijz2*rijz2;

                    const T inv2 = device_rsqrt<T>(rijSquared2 + softSquared);
                    const T factor2 = inv2 * inv2 * inv2;

                    const T ai2 = factor2 * SHm[jBody];
                    accX2 += ai2 * rijx2;
                    accY2 += ai2 * rijy2;
                    accZ2 += ai2 * rijz2;
                }
            __syncthreads();
        }

        if (iBody1 < n_bodies) {
            devAccelerations.ax[iBody1] = accX1;
            devAccelerations.ay[iBody1] = accY1;
            devAccelerations.az[iBody1] = accZ1;
        }
        
        if (iBody2 < n_bodies) {
            devAccelerations.ax[iBody2] = accX2;
            devAccelerations.ay[iBody2] = accY2;
            devAccelerations.az[iBody2] = accZ2;
        }
    }
}
template <typename T> 
const accSoA_t<T>& SimulationNBodyCUDATileFullDevice<T>::getAccSoA() {
    accSoA.ax.resize(this->getBodies()->getN());
    accSoA.ay.resize(this->getBodies()->getN());
    accSoA.az.resize(this->getBodies()->getN());
    
    CUDA_CHECK(cudaMemcpy(accSoA.ax.data(), this->devAccelerations.ax, this->getBodies()->getN() * sizeof(T), cudaMemcpyDeviceToHost));        
    CUDA_CHECK(cudaMemcpy(accSoA.ay.data(), this->devAccelerations.ay, this->getBodies()->getN() * sizeof(T), cudaMemcpyDeviceToHost));     
    CUDA_CHECK(cudaMemcpy(accSoA.az.data(), this->devAccelerations.az, this->getBodies()->getN() * sizeof(T), cudaMemcpyDeviceToHost));        

    return accSoA;
}

template <typename T>
void SimulationNBodyCUDATileFullDevice<T>::computeBodiesAcceleration()
{
    // CUDA_CHECK(cudaFuncSetAttribute(devComputeBodiesAccelerationTileFullDevice<T>, 
    //                      cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    // CUDA_CHECK(cudaGetLastError());
    // devComputeBodiesAccelerationTileFullDevice<T><<<blocks, threads>>>(
    //                                         this->devAccelerations,
    //                                         this->cudaBodiesPtr->getDevDataSoA(),
    //                                         n, this->G, this->softSquared);

    // if ( this->_elem_per_thread == 10000 ) {
        
        // devComputeBodiesAccelerationTileFullDevice_2elem<T><<<this->_num_blocks, this->_num_threads>>>(
        //                                         this->devAccelerations,
        //                                         this->cudaBodiesPtr->getDevDataSoA(),
        //                                         this->bodies->getN(), this->G, this->softSquared,
        //                                         this->_elem_per_thread);

    // } else {
        devComputeBodiesAccelerationTileFullDevice<T><<<this->_num_blocks, this->_num_threads>>>(
                                                this->devAccelerations,
                                                this->cudaBodiesPtr->getDevDataSoA(),
                                                this->devGM,
                                                this->bodies->getN(), this->G, this->softSquared,
                                                this->_elem_per_thread);
    // }

        // int n = (int)this->getBodies()->getN();
        // int threads = 1024; 
        // int blocks = (n + threads - 1) / threads;  
        // devComputeBodiesAccelerationTileFullDevice200k<T><<<blocks,threads>>>(
        //                                         this->devAccelerations,
        //                                         this->cudaBodiesPtr->getDevDataSoA(),
        //                                         this->devGM,
        //                                         this->bodies->getN(), this->G, this->softSquared);

    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
SimulationNBodyCUDATileFullDevice<T>::~SimulationNBodyCUDATileFullDevice() {
    CUDA_CHECK(cudaFree(devAccelerations.ax));
    CUDA_CHECK(cudaFree(devAccelerations.ay));
    CUDA_CHECK(cudaFree(devAccelerations.az));
}

template class SimulationNBodyCUDATileFullDevice<float>;
// template class SimulationNBodyCUDATileFullDevice<double>;

#endif