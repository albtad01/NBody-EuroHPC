#ifndef SIMULATION_N_BODY_CUDA_TILE_FULL_DEVICE_CU_
#define SIMULATION_N_BODY_CUDA_TILE_FULL_DEVICE_CU_

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>

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

template <typename T>
__device__ __forceinline__ T device_rsqrt(T val);

template <>
__device__ __forceinline__ float device_rsqrt<float>(float val) { return rsqrtf(val); }

template <>
__device__ __forceinline__ double device_rsqrt<double>(double val) { return rsqrt(val); }

template <typename T>
SimulationNBodyCUDATileFullDevice<T>::SimulationNBodyCUDATileFullDevice(
        const BodiesAllocatorInterface<T>& allocator, const T soft, const bool transfer_each_iteration)
    : SimulationNBodyInterface<T>(allocator, soft), softSquared{soft*soft}, 
      transfer_each_iteration{transfer_each_iteration}
{
    this->flopsPerIte = 20.f * (T)this->getBodies()->getN() * (T)this->getBodies()->getN();

    CUDA_CHECK(cudaMalloc(&this->devAccelerations.ax, this->getBodies()->getN()*sizeof(T)));
    CUDA_CHECK(cudaMalloc(&this->devAccelerations.ay, this->getBodies()->getN()*sizeof(T)));
    CUDA_CHECK(cudaMalloc(&this->devAccelerations.az, this->getBodies()->getN()*sizeof(T)));

    this->cudaBodiesPtr = std::dynamic_pointer_cast<CUDABodies<T>>(this->bodies);

    if ( this->cudaBodiesPtr == nullptr ) {
        std::cout << "Error in converting to CUDABodies!!!" << std::endl;
    }
}

template <typename T>
__global__ void devInitIterationTileFullDevice(devAccSoA_t<T> devAccelerations,
                                 const int n_bodies) {
    int iBody = blockIdx.x * blockDim.x + threadIdx.x;
    if ( iBody < n_bodies ) {
        devAccelerations.ax[iBody] = T(0);
        devAccelerations.ay[iBody] = T(0);
        devAccelerations.az[iBody] = T(0);
    }
}

template <typename T>
void SimulationNBodyCUDATileFullDevice<T>::initIteration()
{
    int n = (int)this->getBodies()->getN();
    int threads = 256; 
    int blocks = (n + threads - 1) / threads;  
    
    devInitIterationTileFullDevice<T><<<blocks, threads>>>(devAccelerations, n);
    
    CUDA_CHECK(cudaGetLastError());    
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

template <typename T>
__global__ void devComputeBodiesAccelerationTileFullDevice(
    devAccSoA_t<T> devAccelerations, 
    const devDataSoA_t<T> devDataSoA,
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

        if (iBody < n_bodies) {
            
            #pragma unroll 32
            for (int jBody = 0; jBody < TILE_SIZE; jBody++) {
                const T rijx = SHqx[jBody] - rix;
                const T rijy = SHqy[jBody] - riy;
                const T rijz = SHqz[jBody] - riz;

                const T rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;

                const T inv = device_rsqrt<T>(rijSquared + softSquared);
                const T factor = G * (inv * inv * inv);
                
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
    int n = (int)this->getBodies()->getN();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;  

    devComputeBodiesAccelerationTileFullDevice<T><<<blocks, threads>>>(
                                            this->devAccelerations,
                                            this->cudaBodiesPtr->getDevDataSoA(),
                                            n, this->G, this->softSquared);

    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
SimulationNBodyCUDATileFullDevice<T>::~SimulationNBodyCUDATileFullDevice() {
    CUDA_CHECK(cudaFree(devAccelerations.ax));
    CUDA_CHECK(cudaFree(devAccelerations.ay));
    CUDA_CHECK(cudaFree(devAccelerations.az));
}

template class SimulationNBodyCUDATileFullDevice<float>;
template class SimulationNBodyCUDATileFullDevice<double>;

#endif