#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyCUDATileFullDevice.hpp"

SimulationNBodyCUDATileFullDevice::SimulationNBodyCUDATileFullDevice(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface<float>(nBodies, scheme, soft, randInit), softSquared{soft*soft}
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());

    const dataSoA_t<float> &d = this->getBodies().getDataSoA();
    cudaMalloc(&this->devM, this->getBodies().getN()*sizeof(float));
    cudaMemcpy(this->devM, d.m.data(), this->getBodies().getN() * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&this->devQx, this->getBodies().getN()*sizeof(float));
    cudaMalloc(&this->devQy, this->getBodies().getN()*sizeof(float));
    cudaMalloc(&this->devQz, this->getBodies().getN()*sizeof(float));
    cudaMalloc(&this->devAccelerations, this->getBodies().getN()*sizeof(accAoS_t<float>));
}

__global__ void devInitIterationTileFullDevice(accAoS_t<float>* devAccelerations,
                                 int n_bodies) {
    unsigned long iBody = blockIdx.x * blockDim.x + threadIdx.x;
    if ( iBody <= n_bodies ) {
        devAccelerations[iBody].ax = 0.f;
        devAccelerations[iBody].ay = 0.f;
        devAccelerations[iBody].az = 0.f;
    }
}

void SimulationNBodyCUDATileFullDevice::initIteration()
{
    int threads = 1024;
    int blocks = (this->getBodies().getN() + threads - 1) / threads;  
    if ( blocks == 1 ) {
        blocks = 1;
        threads = this->getBodies().getN();
    }
    devInitIterationTileFullDevice<<<blocks, threads>>>(devAccelerations, this->getBodies().getN());
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
    cudaDeviceSynchronize();
}

void SimulationNBodyCUDATileFullDevice::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}

__global__ void devComputeBodiesAccelerationTileFullDevice(
    accAoS_t<float>* devAccelerations,
    float* devM,
    float* devQx,
    float* devQy,
    float* devQz,
    int n_bodies,
    const float G,
    const float softSquared
) {
    // Allocating maximum amount that can fit in shared memory
    constexpr int N = 1;
    constexpr int TILE_SIZE = N * 1024;

    __shared__ float SHm[TILE_SIZE];
    __shared__ float SHqx[TILE_SIZE];
    __shared__ float SHqy[TILE_SIZE];
    __shared__ float SHqz[TILE_SIZE];

    unsigned long iBody = blockIdx.x * blockDim.x + threadIdx.x;

    const float rix = (iBody < n_bodies) ? devQx[iBody] : 0.0f;
    const float riy = (iBody < n_bodies) ? devQy[iBody] : 0.0f;
    const float riz = (iBody < n_bodies) ? devQz[iBody] : 0.0f;

    int sh_idx;
    int gg_idx;

    for (int base_idx = 0; base_idx < n_bodies; base_idx += TILE_SIZE) {

        // TileFullDevice copy
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
                const float rijx = SHqx[jBody] - rix; // 1 flop
                const float rijy = SHqy[jBody] - riy; // 1 flop
                const float rijz = SHqz[jBody] - riz; // 1 flop

                // compute the || rij ||² distance between body i and body j
                const float rijSquared =
                    rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops

                // compute the acceleration value between body i and body j
                // const float partial_denom = std::sqrt(rijSquared + softSquared);
                // const float factor = G / (partial_denom * partial_denom * partial_denom);

                const float partial_factor = rsqrtf(rijSquared + softSquared); // ~ 4 flops
                const float factor =
                    G * (partial_factor * partial_factor * partial_factor); // 3 flops

                const float ai = factor * SHm[jBody]; // 1 flop

                // add the acceleration value into the acceleration vector
                devAccelerations[iBody].ax += ai * rijx; // 2 flops
                devAccelerations[iBody].ay += ai * rijy; // 2 flops
                devAccelerations[iBody].az += ai * rijz; // 2 flops
            }
        }

        __syncthreads();
    }
}

void SimulationNBodyCUDATileFullDevice::computeBodiesAcceleration()
{
    int threads = 1024;
    int blocks = (this->getBodies().getN() + threads - 1) / threads;  
    if ( blocks == 1 ) {
        threads = this->getBodies().getN();
    }

    const dataSoA_t<float> &d = this->getBodies().getDataSoA();
    cudaMemcpy(this->devQx, d.qx.data(), this->getBodies().getN() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->devQy, d.qy.data(), this->getBodies().getN() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->devQz, d.qz.data(), this->getBodies().getN() * sizeof(float), cudaMemcpyHostToDevice);

    devComputeBodiesAccelerationTileFullDevice<<<blocks, threads>>>(
                                            this->devAccelerations,
                                            this->devM,this->devQx,this->devQy,this->devQz,
                                            this->getBodies().getN(), this->G, this->softSquared);
    cudaDeviceSynchronize();

    cudaMemcpy(this->accelerations.data(), this->devAccelerations, 
               this->getBodies().getN() * sizeof(accAoS_t<float>), cudaMemcpyDeviceToHost);
}

SimulationNBodyCUDATileFullDevice::~SimulationNBodyCUDATileFullDevice() {
    cudaFree(devM);
    cudaFree(devQx);
    cudaFree(devQy);
    cudaFree(devQz);
    cudaFree(devAccelerations);
}
