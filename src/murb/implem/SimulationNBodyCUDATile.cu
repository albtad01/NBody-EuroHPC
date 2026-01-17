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
SimulationNBodyCUDATile<T>::SimulationNBodyCUDATile(const BodiesAllocatorInterface<T>& allocator, const T soft)
    : SimulationNBodyInterface<T>(allocator, soft), softSquared{soft*soft}
{
    this->flopsPerIte = 20.f * (T)this->getBodies()->getN() * (T)this->getBodies()->getN();
    this->accelerations.resize(this->getBodies()->getN());

    const dataSoA_t<T> &d = this->getBodies()->getDataSoA();
    cudaMalloc(&this->devM, this->getBodies()->getN()*sizeof(T));
    cudaMemcpy(this->devM, d.m.data(), this->getBodies()->getN() * sizeof(T), cudaMemcpyHostToDevice);

    cudaMalloc(&this->devQx, this->getBodies()->getN()*sizeof(T));
    cudaMalloc(&this->devQy, this->getBodies()->getN()*sizeof(T));
    cudaMalloc(&this->devQz, this->getBodies()->getN()*sizeof(T));
    cudaMalloc(&this->devAccelerations, this->getBodies()->getN()*sizeof(accAoS_t<T>));
}

template <typename T>
__global__ void devInitIteration(accAoS_t<T>* devAccelerations,
                                 int n_bodies) {
    unsigned long iBody = blockIdx.x * blockDim.x + threadIdx.x;
    if ( iBody <= n_bodies ) {
        devAccelerations[iBody].ax = 0.f;
        devAccelerations[iBody].ay = 0.f;
        devAccelerations[iBody].az = 0.f;
    }
}

template <typename T>
void SimulationNBodyCUDATile<T>::initIteration()
{
    int threads = 1024;
    int blocks = (this->getBodies()->getN() + threads - 1) / threads;  
    if ( blocks == 1 ) {
        blocks = 1;
        threads = this->getBodies()->getN();
    }
    devInitIteration<T><<<blocks, threads>>>(devAccelerations, this->getBodies()->getN());
    for (unsigned long iBody = 0; iBody < this->getBodies()->getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
    cudaDeviceSynchronize();
}

template <typename T>
void SimulationNBodyCUDATile<T>::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies->updatePositionsAndVelocities(this->accelerations, this->dt);
}

template <typename T>
__global__ void devComputeBodiesAccelerationTile(
    accAoS_t<T>* devAccelerations,
    T* devM,
    T* devQx,
    T* devQy,
    T* devQz,
    int n_bodies,
    const T G,
    const T softSquared
) {
    // Allocating maximum amount that can fit in shared memory
    constexpr int N = 1;
    constexpr int TILE_SIZE = N * 1024;

    __shared__ T SHm[TILE_SIZE];
    __shared__ T SHqx[TILE_SIZE];
    __shared__ T SHqy[TILE_SIZE];
    __shared__ T SHqz[TILE_SIZE];

    unsigned long iBody = blockIdx.x * blockDim.x + threadIdx.x;

    const T rix = (iBody < n_bodies) ? devQx[iBody] : 0.0f;
    const T riy = (iBody < n_bodies) ? devQy[iBody] : 0.0f;
    const T riz = (iBody < n_bodies) ? devQz[iBody] : 0.0f;

    int sh_idx;
    int gg_idx;

    for (int base_idx = 0; base_idx < n_bodies; base_idx += TILE_SIZE) {

        // Tile copy
        for (int i = 0; i < N; i++) {
            sh_idx = threadIdx.x + i * 1024;
            gg_idx = base_idx + sh_idx;

            if (gg_idx >= n_bodies) {
                SHm[sh_idx]  = 0.0;
                SHqx[sh_idx] = 0.0;
                SHqy[sh_idx] = 0.0;
                SHqz[sh_idx] = 0.0;
            } else {
                SHm[sh_idx]  = devM[gg_idx];
                SHqx[sh_idx] = devQx[gg_idx];
                SHqy[sh_idx] = devQy[gg_idx];
                SHqz[sh_idx] = devQz[gg_idx];
            }
        }

        __syncthreads();

        if (iBody < n_bodies) {
            int tile_end = min(TILE_SIZE, n_bodies - base_idx);

            for (int jBody = 0; jBody < tile_end; jBody++) {
                const T rijx = SHqx[jBody] - rix; // 1 flop
                const T rijy = SHqy[jBody] - riy; // 1 flop
                const T rijz = SHqz[jBody] - riz; // 1 flop

                // compute the || rij ||² distance between body i and body j
                const T rijSquared =
                    rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops

                // compute the acceleration value between body i and body j
                // const T partial_denom = std::sqrt(rijSquared + softSquared);
                // const T factor = G / (partial_denom * partial_denom * partial_denom);

                const T partial_factor = rsqrtf(rijSquared + softSquared); // ~ 4 flops
                const T factor =
                    G * (partial_factor * partial_factor * partial_factor); // 3 flops

                const T ai = factor * SHm[jBody]; // 1 flop

                // add the acceleration value into the acceleration vector
                devAccelerations[iBody].ax += ai * rijx; // 2 flops
                devAccelerations[iBody].ay += ai * rijy; // 2 flops
                devAccelerations[iBody].az += ai * rijz; // 2 flops
            }
        }

        __syncthreads();
    }
}

template <typename T>
void SimulationNBodyCUDATile<T>::computeBodiesAcceleration()
{
    int threads = 1024;
    int blocks = (this->getBodies()->getN() + threads - 1) / threads;  
    if ( blocks == 1 ) {
        threads = this->getBodies()->getN();
    }

    const dataSoA_t<T> &d = this->getBodies()->getDataSoA();
    cudaMemcpy(this->devQx, d.qx.data(), this->getBodies()->getN() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(this->devQy, d.qy.data(), this->getBodies()->getN() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(this->devQz, d.qz.data(), this->getBodies()->getN() * sizeof(T), cudaMemcpyHostToDevice);

    devComputeBodiesAccelerationTile<T><<<blocks, threads>>>(
                                            this->devAccelerations,
                                            this->devM,this->devQx,this->devQy,this->devQz,
                                            this->getBodies()->getN(), this->G, this->softSquared);
    cudaDeviceSynchronize();

    cudaMemcpy(this->accelerations.data(), this->devAccelerations, 
               this->getBodies()->getN() * sizeof(accAoS_t<T>), cudaMemcpyDeviceToHost);
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