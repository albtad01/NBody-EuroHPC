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
SimulationNBodyCUDATileFullDevice<T>::SimulationNBodyCUDATileFullDevice(
        const BodiesAllocatorInterface<T>& allocator, const T soft)
    : SimulationNBodyInterface<T>(allocator, soft), softSquared{soft*soft}
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
    unsigned long iBody = blockIdx.x * blockDim.x + threadIdx.x;
    if ( iBody < n_bodies ) {
        devAccelerations.ax[iBody] = 0.f;
        devAccelerations.ay[iBody] = 0.f;
        devAccelerations.az[iBody] = 0.f;
    }
}

template <typename T>
void SimulationNBodyCUDATileFullDevice<T>::initIteration()
{
    int threads = 1024;
    int blocks = (this->getBodies()->getN() + threads - 1) / threads;  
    if ( blocks == 1 ) {
        blocks = 1;
        threads = this->getBodies()->getN();
    }
    devInitIterationTileFullDevice<T><<<blocks, threads>>>(devAccelerations, 
                                                        this->getBodies()->getN());
    // for (unsigned long iBody = 0; iBody < this->getBodies()->getN(); iBody++) {
    //     this->accelerations[iBody].ax = 0.f;
    //     this->accelerations[iBody].ay = 0.f;
    //     this->accelerations[iBody].az = 0.f;
    // }
    CUDA_CHECK(cudaGetLastError());    
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename T>
void SimulationNBodyCUDATileFullDevice<T>::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->cudaBodiesPtr->updatePositionsAndVelocitiesOnDevice(this->devAccelerations, this->dt);
}

template <typename T>
__global__ void devComputeBodiesAccelerationTileFullDevice(
    devAccSoA_t<T> devAccelerations, 
    const devDataSoA_t<T> devDataSoA,
    const int n_bodies,
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

    const T rix = (iBody < n_bodies) ? devDataSoA.qx[iBody] : 0.0f;
    const T riy = (iBody < n_bodies) ? devDataSoA.qy[iBody] : 0.0f;
    const T riz = (iBody < n_bodies) ? devDataSoA.qz[iBody] : 0.0f;

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
                SHm[sh_idx]  = devDataSoA.m[gg_idx];
                SHqx[sh_idx] = devDataSoA.qx[gg_idx];
                SHqy[sh_idx] = devDataSoA.qy[gg_idx];
                SHqz[sh_idx] = devDataSoA.qz[gg_idx];
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
                devAccelerations.ax[iBody] += ai * rijx; // 2 flops
                devAccelerations.ay[iBody] += ai * rijy; // 2 flops
                devAccelerations.az[iBody] += ai * rijz; // 2 flops
            }
        }

        __syncthreads();
    }
}

template <typename T> 
const accSoA_t<T>& SimulationNBodyCUDATileFullDevice<T>::getAccSoA() {
    accSoA.ax.resize(this->getBodies()->getN());
    accSoA.ay.resize(this->getBodies()->getN());
    accSoA.az.resize(this->getBodies()->getN());
    cudaMemcpy(accSoA.ax.data(), this->devAccelerations.ax, this->getBodies()->getN() * sizeof(T), cudaMemcpyDeviceToHost);        
    cudaMemcpy(accSoA.ay.data(), this->devAccelerations.ay, this->getBodies()->getN() * sizeof(T), cudaMemcpyDeviceToHost);     
    cudaMemcpy(accSoA.az.data(), this->devAccelerations.az, this->getBodies()->getN() * sizeof(T), cudaMemcpyDeviceToHost);        

    return accSoA;
}

template <typename T>
void SimulationNBodyCUDATileFullDevice<T>::computeBodiesAcceleration()
{
    static int count = 0;
    int threads = 1024;
    int blocks = (this->getBodies()->getN() + threads - 1) / threads;  
    if ( blocks == 1 ) {
        threads = this->getBodies()->getN();
    }
    devComputeBodiesAccelerationTileFullDevice<T><<<blocks, threads>>>(
                                            this->devAccelerations,
                                            this->cudaBodiesPtr->getDevDataSoA(),
                                            this->getBodies()->getN(), this->G, this->softSquared);


    CUDA_CHECK(cudaGetLastError());    
    CUDA_CHECK(cudaDeviceSynchronize());



    count++;
}

template <typename T>
SimulationNBodyCUDATileFullDevice<T>::~SimulationNBodyCUDATileFullDevice() {
    CUDA_CHECK(cudaFree(devAccelerations.ax));
    CUDA_CHECK(cudaFree(devAccelerations.ay));
    CUDA_CHECK(cudaFree(devAccelerations.az));
}

template class SimulationNBodyCUDATileFullDevice<float>;
template class SimulationNBodyCUDATileFullDevice<double>;