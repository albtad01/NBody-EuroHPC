#ifndef SIMULATION_N_BODY_CUDA_TILE_CU_
#define SIMULATION_N_BODY_CUDA_TILE_CU_

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyCUDATile.hpp"

template <typename T>
__device__ __forceinline__ T device_rsqrt(T val);

template <>
__device__ __forceinline__ float device_rsqrt<float>(float val) { return rsqrtf(val); }

template <>
__device__ __forceinline__ double device_rsqrt<double>(double val) { return rsqrt(val); }

template <typename T>
SimulationNBodyCUDATile<T>::SimulationNBodyCUDATile(const BodiesAllocatorInterface<T>& allocator, const T soft)
    : SimulationNBodyInterface<T>(allocator, soft), softSquared{soft*soft}
{
    this->flopsPerIte = 20.f * (T)this->getBodies()->getN() * (T)this->getBodies()->getN();
    this->accelerations.resize(this->getBodies()->getN());

    const dataSoA_t<T> &d = this->getBodies()->getDataSoA();
    cudaMalloc(&this->devM,  this->getBodies()->getN() * sizeof(T));
    cudaMemcpy(this->devM, d.m.data(), this->getBodies()->getN() * sizeof(T), cudaMemcpyHostToDevice);

    cudaMalloc(&this->devQx, this->getBodies()->getN() * sizeof(T));
    cudaMalloc(&this->devQy, this->getBodies()->getN() * sizeof(T));
    cudaMalloc(&this->devQz, this->getBodies()->getN() * sizeof(T));
    cudaMalloc(&this->devAccelerations, this->getBodies()->getN() * sizeof(accAoS_t<T>));
}

template <typename T>
__global__ void devInitIteration(accAoS_t<T>* devAccelerations, int n_bodies)
{
    int iBody = blockIdx.x * blockDim.x + threadIdx.x;
    if (iBody < n_bodies) {
        devAccelerations[iBody].ax = T(0);
        devAccelerations[iBody].ay = T(0);
        devAccelerations[iBody].az = T(0);
    }
}

template <typename T>
void SimulationNBodyCUDATile<T>::initIteration()
{
    int n = (int)this->getBodies()->getN();
    int threads = 256;                      
    int blocks  = (n + threads - 1) / threads;

    devInitIteration<T><<<blocks, threads>>>(this->devAccelerations, n);

}

template <typename T>
__global__ void devComputeBodiesAccelerationTile(
    accAoS_t<T>* devAccelerations,
    const T* __restrict__ devM,
    const T* __restrict__ devQx,
    const T* __restrict__ devQy,
    const T* __restrict__ devQz,
    int n_bodies,
    const T G,
    const T softSquared
){
    constexpr int N = 1;
    constexpr int TILE_SIZE = N * 1024;

    __shared__ T SHm[TILE_SIZE];
    __shared__ T SHqx[TILE_SIZE];
    __shared__ T SHqy[TILE_SIZE];
    __shared__ T SHqz[TILE_SIZE];

    int iBody = blockIdx.x * blockDim.x + threadIdx.x;

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
        devAccelerations[iBody].ax = accX;
        devAccelerations[iBody].ay = accY;
        devAccelerations[iBody].az = accZ;
    }
}

template <typename T>
void SimulationNBodyCUDATile<T>::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    this->bodies->updatePositionsAndVelocities(this->accelerations, this->dt);
}

template <typename T>
void SimulationNBodyCUDATile<T>::computeBodiesAcceleration()
{
    int n = (int)this->getBodies()->getN();
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;

    const dataSoA_t<T> &d = this->getBodies()->getDataSoA();
    cudaMemcpy(this->devQx, d.qx.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(this->devQy, d.qy.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(this->devQz, d.qz.data(), n * sizeof(T), cudaMemcpyHostToDevice);

    devComputeBodiesAccelerationTile<T><<<blocks, threads>>>(
        this->devAccelerations,
        this->devM, this->devQx, this->devQy, this->devQz,
        n, this->G, this->softSquared
    );
    cudaMemcpy(this->accelerations.data(), this->devAccelerations,
               n * sizeof(accAoS_t<T>), cudaMemcpyDeviceToHost);
}

template <typename T>
SimulationNBodyCUDATile<T>::~SimulationNBodyCUDATile() {
    cudaFree(devM);
    cudaFree(devQx);
    cudaFree(devQy);
    cudaFree(devQz);
    cudaFree(devAccelerations);
}

template class SimulationNBodyCUDATile<float>;
template class SimulationNBodyCUDATile<double>;

#endif