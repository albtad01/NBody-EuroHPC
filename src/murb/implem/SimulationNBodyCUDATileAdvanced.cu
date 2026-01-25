#ifndef SIMULATION_N_BODY_CUDA_TILE_ADVANCED_CU_
#define SIMULATION_N_BODY_CUDA_TILE_ADVANCED_CU_

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <chrono>

#include "SimulationNBodyCUDATileAdvanced.hpp"

template <
    class result_t   = std::chrono::nanoseconds,
    class clock_t    = std::chrono::steady_clock,
    class duration_t = std::chrono::nanoseconds
>
result_t since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}

template <typename T>
__device__ __forceinline__ T device_rsqrt(T val);

template <>
__device__ __forceinline__ float device_rsqrt<float>(float val) { return rsqrtf(val); }

template <>
__device__ __forceinline__ double device_rsqrt<double>(double val) { return rsqrt(val); }


template <typename T>
__device__ __forceinline__ T device_fma(T x, T y, T z);

template <>
__device__ __forceinline__ float device_fma<float>(float x, float y, float z) { return fmaf(x,y,z); }

template <>
__device__ __forceinline__ double device_fma<double>(double x, double y, double z) { return fma(x,y,z); }

template <typename T>
SimulationNBodyCUDATileAdvanced<T>::SimulationNBodyCUDATileAdvanced(const BodiesAllocatorInterface<T>& allocator, const T soft)
    : SimulationNBodyInterface<T>(allocator, soft), softSquared{soft*soft}
{
    this->flopsPerIte = 20.f * (T)this->getBodies()->getN() * (T)this->getBodies()->getN();
    this->accelerations.ax.resize(this->getBodies()->getN());
    this->accelerations.ax.resize(this->getBodies()->getN());
    this->accelerations.ax.resize(this->getBodies()->getN());

    const int NUM_SM = 128;
    int n = (int)this->getBodies()->getN();
    if ( n <= 128 * 256 ) {
        this->_num_threads = 256;
    } else if ( n <= 128 * 512 ) {
        this->_num_threads = 512;
    } else {
        this->_num_threads = 1024;
    }

    this->_num_blocks = std::min(NUM_SM, (n + this->_num_threads - 1) / this->_num_threads);
    int total_threads = this->_num_blocks * this->_num_threads;
    this->_elem_per_thread = (n + total_threads - 1) / total_threads;

    // std::cout << "Number of threads: " << _num_threads << std::endl;
    // std::cout << "Number of blocks: " << _num_blocks << std::endl;
    // std::cout << "Element per thread: " << _elem_per_thread << std::endl;
    // std::cout << "Total desired bodies: " << total_threads * this->_elem_per_thread << std::endl;


    const dataSoA_t<T> &d = this->getBodies()->getDataSoA();
    cudaMalloc(&this->devM,  this->getBodies()->getN() * sizeof(T));
    cudaMemcpy(this->devM, d.m.data(), this->getBodies()->getN() * sizeof(T), cudaMemcpyHostToDevice);

    cudaMalloc(&this->devQx, this->getBodies()->getN() * sizeof(T));
    cudaMalloc(&this->devQy, this->getBodies()->getN() * sizeof(T));
    cudaMalloc(&this->devQz, this->getBodies()->getN() * sizeof(T));
    cudaMalloc(&this->devAccelerations.x, this->getBodies()->getN() * sizeof(T));
    cudaMalloc(&this->devAccelerations.y, this->getBodies()->getN() * sizeof(T));
    cudaMalloc(&this->devAccelerations.z, this->getBodies()->getN() * sizeof(T));

}

// template <typename T>
// __global__ void devInitIteration(
//     accAoS_t<T>* devAccelerations, 
//     int n_bodies,
//     const int elem_per_thread
// )
// {
//     for (int k = 0; k < elem_per_thread; k++) {
//         int iBody = blockIdx.x * blockDim.x * elem_per_thread + threadIdx.x + k * blockDim.x;
//         if (iBody < n_bodies) {
//             devAccelerations[iBody].x = T(0);
//             devAccelerations[iBody].y = T(0);
//             devAccelerations[iBody].z = T(0);
//         }
//     }
// }

template <typename T>
void SimulationNBodyCUDATileAdvanced<T>::initIteration()
{
    // int n = (int)this->getBodies()->getN();
    // devInitIteration<T><<<this->_num_blocks, this->_num_threads>>>(
    //     this->devAccelerations, n, this->_elem_per_thread);
    // cudaDeviceSynchronize();
}

template <typename T>
__global__ void devComputeBodiesAccelerationTile(
    devAccSoA_t<T> devAccelerations,
    const T* __restrict__ devM,
    const T* __restrict__ devQx,
    const T* __restrict__ devQy,
    const T* __restrict__ devQz,
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
            rix = devQx[iBody];
            riy = devQy[iBody];
            riz = devQz[iBody];
        }

        for (int base_idx = 0; base_idx < n_bodies; base_idx += TILE_SIZE) {

            for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
                int gg_idx = base_idx + i;
                if (gg_idx < n_bodies) {
                    SHm[i]  = devM[gg_idx];
                    SHqx[i] = devQx[gg_idx];
                    SHqy[i] = devQy[gg_idx];
                    SHqz[i] = devQz[gg_idx];
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
                    // const T factor = inv * inv * inv;
                    // T ai = G * SHm[jBody];
                    // ai = ai * factor;
                    const T factor = G * (inv * inv * inv);
                    const T ai = factor * SHm[jBody];

                    accX += ai * rijx;
                    accY += ai * rijy;
                    accZ += ai * rijz;
                    // const T ai = G * (inv * inv * inv) * SHm[jBody];
                    // accX = device_fma<T>(ai, rijx, accX);
                    // accY = device_fma<T>(ai, rijy, accY);
                    // accZ = device_fma<T>(ai, rijz, accZ);
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
}

template <typename T>
void SimulationNBodyCUDATileAdvanced<T>::computeOneIteration()
{
    this->initIteration();

    auto start0 = std::chrono::steady_clock::now();
    this->computeBodiesAcceleration();
    // std::cout << "TOTAL COMPUTE: Elapsed(ns)=" << since(start0).count() << std::endl;
    auto start1 = std::chrono::steady_clock::now();
    this->bodies->updatePositionsAndVelocities(this->accelerations, this->dt);
    // std::cout << "UPDATE: Elapsed(ns)=" << since(start1).count() << std::endl;
}

template <typename T>
void SimulationNBodyCUDATileAdvanced<T>::computeBodiesAcceleration()
{
    int n = (int)this->getBodies()->getN();
    const dataSoA_t<T> &d = this->getBodies()->getDataSoA();

    auto start0 = std::chrono::steady_clock::now();
    cudaMemcpy(this->devQx, d.qx.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(this->devQy, d.qy.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(this->devQz, d.qz.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    // std::cout << "MEMCPY pos: Elapsed(ns)=" << since(start0).count() << std::endl;

    auto start1 = std::chrono::steady_clock::now();
    devComputeBodiesAccelerationTile<T><<<this->_num_blocks, this->_num_threads>>>(
        this->devAccelerations,
        this->devM, this->devQx, this->devQy, this->devQz,
        n, this->G, this->softSquared, this->_elem_per_thread
    );
    cudaDeviceSynchronize();
    // std::cout << "KERNEL: Elapsed(ns)=" << since(start1).count() << std::endl;
    
    auto start2 = std::chrono::steady_clock::now();
    cudaMemcpy(this->accelerations.ax.data(), this->devAccelerations.x,
               n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->accelerations.ay.data(), this->devAccelerations.y,
               n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->accelerations.az.data(), this->devAccelerations.z,
               n * sizeof(T), cudaMemcpyDeviceToHost);
    // std::cout << "MEMCPY acc: Elapsed(ns)=" << since(start2).count() << std::endl;
}

template <typename T>
SimulationNBodyCUDATileAdvanced<T>::~SimulationNBodyCUDATileAdvanced() {
    cudaFree(devM);
    cudaFree(devQx);
    cudaFree(devQy);
    cudaFree(devQz);
    cudaFree(devAccelerations.x);
    cudaFree(devAccelerations.y);
    cudaFree(devAccelerations.z);
}

template class SimulationNBodyCUDATileAdvanced<float>;
template class SimulationNBodyCUDATileAdvanced<double>;

#endif