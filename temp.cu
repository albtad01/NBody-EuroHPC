#ifndef SIMULATION_N_BODY_CUDA_TILE_CU_
#define SIMULATION_N_BODY_CUDA_TILE_CU_

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <algorithm>

#include <cuda_runtime.h>

#include "SimulationNBodyCUDATile.hpp"

// --------------------------
// Fast rsqrt (float/double)
// --------------------------
template <typename T>
__device__ __forceinline__ T device_rsqrt(T val);

template <>
__device__ __forceinline__ float device_rsqrt<float>(float val) { return rsqrtf(val); }

template <>
__device__ __forceinline__ double device_rsqrt<double>(double val) { return rsqrt(val); }

// --------------------------
// Helper: cap blocks using SM count
// --------------------------
inline int cappedBlocksForGPU(int blocksNeeded, int threadsPerBlock, int blocksPerSM = 4, int device = 0)
{
    (void)threadsPerBlock; // not used right now, but kept for future tuning
    int sm = 0;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, device);
    int cap = std::max(1, sm * blocksPerSM);
    return std::max(1, std::min(blocksNeeded, cap));
}

// --------------------------
// ctor/dtor
// --------------------------
template <typename T>
SimulationNBodyCUDATile<T>::SimulationNBodyCUDATile(const BodiesAllocatorInterface<T>& allocator, const T soft)
    : SimulationNBodyInterface<T>(allocator, soft), softSquared{soft * soft}
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
SimulationNBodyCUDATile<T>::~SimulationNBodyCUDATile()
{
    cudaFree(devM);
    cudaFree(devQx);
    cudaFree(devQy);
    cudaFree(devQz);
    cudaFree(devAccelerations);
}

// --------------------------
// initIteration kernel (grid-stride: each thread zeros multiple bodies)
// --------------------------
template <typename T>
__global__ void devInitIteration(accAoS_t<T>* devAccelerations, int n_bodies)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int iBody = tid; iBody < n_bodies; iBody += stride) {
        devAccelerations[iBody].ax = T(0);
        devAccelerations[iBody].ay = T(0);
        devAccelerations[iBody].az = T(0);
    }
}

template <typename T>
void SimulationNBodyCUDATile<T>::initIteration()
{
    const int n = (int)this->getBodies()->getN();

    // Reasonable defaults on RTX 4090:
    // - 256 or 512 often best
    // - 1024 sometimes hurts occupancy due to registers/blocks-per-SM limits
    const int threads = 256;          // try also 512
    const int blocksNeeded = (n + threads - 1) / threads;

    // RTX 4090 has 128 SMs; cap blocks to avoid massive grids when n grows
    const int blocks = cappedBlocksForGPU(blocksNeeded, threads, /*blocksPerSM=*/4);

    devInitIteration<T><<<blocks, threads>>>(this->devAccelerations, n);
}

// --------------------------
// Acceleration kernel (grid-stride over iBody: each thread handles multiple bodies)
// Tiling over j bodies remains the same
// --------------------------
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
    // Tile size for the "j bodies" dimension
    // Keep at 1024 to match your original approach; adjust if you tune later.
    constexpr int TILE_SIZE = 1024;

    __shared__ T SHm[TILE_SIZE];
    __shared__ T SHqx[TILE_SIZE];
    __shared__ T SHqy[TILE_SIZE];
    __shared__ T SHqz[TILE_SIZE];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int strideBodies = blockDim.x * gridDim.x;

    // Each thread processes multiple iBody
    for (int iBody = tid; iBody < n_bodies; iBody += strideBodies) {

        const T rix = devQx[iBody];
        const T riy = devQy[iBody];
        const T riz = devQz[iBody];

        T accX = T(0), accY = T(0), accZ = T(0);

        // Sweep all bodies in tiles
        for (int base_idx = 0; base_idx < n_bodies; base_idx += TILE_SIZE) {

            // Cooperative load of tile into shared memory
            for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
                const int gg_idx = base_idx + i;
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

            // Compute contribution of this tile
            // Unroll factor: don't overdo it; 8 is usually safe.
            #pragma unroll 8
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

            __syncthreads();
        }

        devAccelerations[iBody].ax = accX;
        devAccelerations[iBody].ay = accY;
        devAccelerations[iBody].az = accZ;
    }
}

// --------------------------
// main iteration
// --------------------------
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
    const int n = (int)this->getBodies()->getN();

    // Reasonable defaults on RTX 4090:
    const int threads = 256;                 // try also 512
    const int blocksNeeded = (n + threads - 1) / threads;
    const int blocks = cappedBlocksForGPU(blocksNeeded, threads, /*blocksPerSM=*/4);

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

// explicit instantiation
template class SimulationNBodyCUDATile<float>;
template class SimulationNBodyCUDATile<double>;

#endif // SIMULATION_N_BODY_CUDA_TILE_CU_