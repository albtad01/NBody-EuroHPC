#ifndef SIMULATION_N_BODY_CUDA_TILE_FULL_DEVICE_200K_CU_
#define SIMULATION_N_BODY_CUDA_TILE_FULL_DEVICE_200K_CU_

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>

#include <cuda_runtime.h>

#include "SimulationNBodyCUDATileFullDevice200k.hpp"

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
__global__ void devInitializeDevGM(const devDataSoA_t<T> devDataSoA, const int n_bodies, T G, T* devGM) {
    unsigned long iBody = blockIdx.x * blockDim.x + threadIdx.x;
    if ( iBody < n_bodies ) {
        devGM[iBody] = G * devDataSoA.m[iBody];
    }
}

template <typename T>
SimulationNBodyCUDATileFullDevice200k<T>::SimulationNBodyCUDATileFullDevice200k(
        const BodiesAllocatorInterface<T>& allocator, const T soft, const bool transfer_each_iteration)
    : SimulationNBodyInterface<T>(allocator, soft), softSquared{soft*soft}, 
      transfer_each_iteration{transfer_each_iteration}
{
    this->flopsPerIte = 20.f * (T)this->getBodies()->getN() * (T)this->getBodies()->getN();

    int n = (int)this->getBodies()->getN();

    this->_num_threads = 1024;

    this->_num_blocks = (this->getBodies()->getN() + _num_threads - 1) / _num_threads;  
    if ( _num_blocks == 1 ) {
        _num_threads = this->getBodies()->getN();
    }

    CUDA_CHECK(cudaMalloc(&this->devAccelerations.x, this->getBodies()->getN()*sizeof(T)));
    CUDA_CHECK(cudaMalloc(&this->devAccelerations.y, this->getBodies()->getN()*sizeof(T)));
    CUDA_CHECK(cudaMalloc(&this->devAccelerations.z, this->getBodies()->getN()*sizeof(T)));
    CUDA_CHECK(cudaMalloc(&this->devGM, this->getBodies()->getN()*sizeof(T)));

    // std::cout << "Number of threads: " << _num_threads << std::endl;
    // std::cout << "Number of blocks: " << _num_blocks << std::endl;
    // std::cout << "Element per thread: " << _elem_per_thread << std::endl;
    // std::cout << "Total desired bodies: " << total_threads * this->_elem_per_thread << std::endl;

    this->cudaBodiesPtr = std::dynamic_pointer_cast<CUDABodies<T>>(this->bodies);

    if ( this->cudaBodiesPtr == nullptr ) {
        std::cout << "Error in converting to CUDABodies!!!" << std::endl;
    }

    devInitializeDevGM<T><<<this->_num_blocks,this->_num_threads>>>(this->cudaBodiesPtr->getDevDataSoA(), 
        this->bodies->getN(),this->G,this->devGM);
}

template <typename T>
void SimulationNBodyCUDATileFullDevice200k<T>::initIteration()
{
}

template <typename T>
void SimulationNBodyCUDATileFullDevice200k<T>::computeOneIteration()
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
        devAccelerations.x[iBody] = accX;
        devAccelerations.y[iBody] = accY;
        devAccelerations.z[iBody] = accZ;
    }
}


template <typename T> 
const accSoA_t<T>& SimulationNBodyCUDATileFullDevice200k<T>::getAccSoA() {
    accSoA.ax.resize(this->getBodies()->getN());
    accSoA.ay.resize(this->getBodies()->getN());
    accSoA.az.resize(this->getBodies()->getN());
    
    CUDA_CHECK(cudaMemcpy(accSoA.ax.data(), this->devAccelerations.x, this->getBodies()->getN() * sizeof(T), cudaMemcpyDeviceToHost));        
    CUDA_CHECK(cudaMemcpy(accSoA.ay.data(), this->devAccelerations.y, this->getBodies()->getN() * sizeof(T), cudaMemcpyDeviceToHost));     
    CUDA_CHECK(cudaMemcpy(accSoA.az.data(), this->devAccelerations.z, this->getBodies()->getN() * sizeof(T), cudaMemcpyDeviceToHost));        

    return accSoA;
}

template <typename T>
void SimulationNBodyCUDATileFullDevice200k<T>::computeBodiesAcceleration()
{
    devComputeBodiesAccelerationTileFullDevice200k<T><<<this->_num_blocks, this->_num_threads>>>(
                                            this->devAccelerations,
                                            this->cudaBodiesPtr->getDevDataSoA(),
                                            this->devGM,
                                            this->bodies->getN(), this->G, this->softSquared);

    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
SimulationNBodyCUDATileFullDevice200k<T>::~SimulationNBodyCUDATileFullDevice200k() {
    CUDA_CHECK(cudaFree(devAccelerations.x));
    CUDA_CHECK(cudaFree(devAccelerations.y));
    CUDA_CHECK(cudaFree(devAccelerations.z));
}

template class SimulationNBodyCUDATileFullDevice200k<float>;
// template class SimulationNBodyCUDATileFullDevice200k<double>;

#endif