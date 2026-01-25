#ifndef SIMULATION_N_BODY_CUDA_LEAPFROG_CU_
#define SIMULATION_N_BODY_CUDA_LEAPFROG_CU_

//#define COMPUTE_ALL_METRICS 
#define COMPUTE_ENERGY_METRIC
// #define COMPUTE_ANGMOMENTUM_METRIC
// #define COMPUTE_DENSITY_CENTER_METRIC

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "SimulationNBodyCUDALeapfrog.hpp"

#define CUDA_CHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        printf("CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
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

template <typename T, typename Q>
SimulationNBodyCUDALeapfrog<T,Q>::SimulationNBodyCUDALeapfrog(
        const BodiesAllocatorInterface<T>& allocator,  
        std::shared_ptr<GPUSimulationHistory<Q>> history, 
        int numIterations,
        const T soft, const bool transfer_each_iteration)
    : SimulationNBodyInterface<T>(allocator, soft), softSquared{soft*soft}, 
      GPUHistoryTrackingInterface<Q>(history),
      numIterations{numIterations},
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

    CUDA_CHECK(cudaMalloc(&this->devAccelerations.x, this->getBodies()->getN()*sizeof(T)));
    CUDA_CHECK(cudaMalloc(&this->devAccelerations.y, this->getBodies()->getN()*sizeof(T)));
    CUDA_CHECK(cudaMalloc(&this->devAccelerations.z, this->getBodies()->getN()*sizeof(T)));
    CUDA_CHECK(cudaMalloc(&this->devGM, this->getBodies()->getN()*sizeof(T)));

    CUDA_CHECK(cudaMalloc(&this->bufferForEnergy, this->getBodies()->getN()*sizeof(T)));

    this->cudaBodiesPtr = std::dynamic_pointer_cast<CUDABodies<T>>(this->bodies);

    if ( this->cudaBodiesPtr == nullptr ) {
        std::cout << "Error in converting to CUDABodies!!!" << std::endl;
    }

    devInitializeDevGM<T><<<this->_num_blocks,this->_num_threads>>>(this->cudaBodiesPtr->getDevDataSoA(), 
        this->bodies->getN(),this->G,this->devGM,this->_elem_per_thread);

    const auto& dataSoA = this->getBodies()->getDataSoA();
    double potential_energy = 0;
    double kinetik_energy = 0;
    for (int i=0;i<this->getBodies()->getN(); i++) {
        for (int j=0; j<i; j++) {
            T rijx = dataSoA.qx[i] - dataSoA.qx[j];
            T rijy = dataSoA.qy[i] - dataSoA.qy[j];
            T rijz = dataSoA.qz[i] - dataSoA.qz[j];
            T rijSquared = rijx*rijx + rijy*rijy + rijz*rijz;
            potential_energy -= this->G * dataSoA.m[i] * dataSoA.m[j] / (std::sqrt(rijSquared+softSquared));
        }
        kinetik_energy += 0.5 * dataSoA.m[i] * (dataSoA.vx[i]*dataSoA.vx[i] + dataSoA.vy[i]*dataSoA.vy[i] + dataSoA.vz[i]*dataSoA.vz[i]);
    }

}


template <typename T, typename Q>
void SimulationNBodyCUDALeapfrog<T,Q>::initIteration()
{
}

template <typename T, typename Q>
void SimulationNBodyCUDALeapfrog<T,Q>::computeOneIteration()
{
    this->currentIteration++; // initially currentIteration = -1 so first iteration is 0
                              // now currentIteration = n

    //this->initIteration();

    // computes a_n
    this->computeBodiesAcceleration();  
    // computes v_{n+1/2} and x_{n+1}
    this->cudaBodiesPtr->updatePositionsAndVelocitiesLeapfrogOnDevice(this->devAccelerations, 
                                        this->dt, this->currentIteration, this->numIterations); 
                                        
    if ( this->transfer_each_iteration ) {
        this->cudaBodiesPtr->getDataSoA();
    }
    // this->computeMetrics();  // computes metrics for previous iteration, starting from initial metrics,
    //                          // since only now v_n is available
    //                          // In other words, x_{n+1} is not used here
    // this->history->copyFromDevice();
}

template <typename T>
__global__ void devComputeBodiesAccelerationLeapfrog(
    devAccSoA_t<T> devAccelerations, 
    const devDataSoA_t<T> devDataSoA,
    const devAccSoA_t<T> devPositionsBuffer,
    const T* devGM,
    const int n_bodies,
    const T softSquared,
    const int elem_per_thread
){
    constexpr int N = 1;
    constexpr int TILE_SIZE = N * 1024;

    __shared__ T SHmg[TILE_SIZE];
    __shared__ T SHqx[TILE_SIZE];
    __shared__ T SHqy[TILE_SIZE];
    __shared__ T SHqz[TILE_SIZE];

    for (int k = 0; k < elem_per_thread; k++) {
        const int iBody =
            blockIdx.x * blockDim.x * elem_per_thread
          + threadIdx.x
          + k * blockDim.x;

        T accX = T(0), accY = T(0), accZ = T(0);
        T rix = T(0), riy = T(0), riz = T(0);

        if (iBody < n_bodies) {
            rix = devPositionsBuffer.x[iBody];
            riy = devPositionsBuffer.y[iBody];
            riz = devPositionsBuffer.z[iBody];
        }

        for (int base_idx = 0; base_idx < n_bodies; base_idx += TILE_SIZE) {

            for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
                int gg_idx = base_idx + i;
                if (gg_idx < n_bodies) {
                    SHmg[i] = devGM[gg_idx];
                    SHqx[i] = devPositionsBuffer.x[gg_idx];
                    SHqy[i] = devPositionsBuffer.y[gg_idx];
                    SHqz[i] = devPositionsBuffer.z[gg_idx];
                } else {
                    SHmg[i] = T(0);
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

                    const T rijSquared =
                        rijx*rijx + rijy*rijy + rijz*rijz;

                    const T inv =
                        device_rsqrt<T>(rijSquared + softSquared);

                    const T ai =
                        inv * inv * inv * SHmg[jBody];

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
}

template <typename T, typename Q=T>
__global__ void devComputeBodiesMetrics(
    const devDataSoA_t<T> devDataSoA,
    const T* devGM,
    const int n_bodies,
    const T softSquared,
    const int elem_per_thread,
    Q* bufferForEnergy
){
    constexpr int N = 1;
    constexpr int TILE_SIZE = N * 1024;

    __shared__ T SHmg[TILE_SIZE];
    __shared__ T SHqx[TILE_SIZE];
    __shared__ T SHqy[TILE_SIZE];
    __shared__ T SHqz[TILE_SIZE];

    for (int k = 0; k < elem_per_thread; k++) {
        const int iBody =
            blockIdx.x * blockDim.x * elem_per_thread
          + threadIdx.x
          + k * blockDim.x;

        Q potential_energy = Q(0);
        Q kinetik_energy   = Q(0);

        T rix = T(0), riy = T(0), riz = T(0);
        T iBodyMass = T(0);

        if (iBody < n_bodies) {
            iBodyMass = devDataSoA.m[iBody];
            rix = devDataSoA.qx[iBody];
            riy = devDataSoA.qy[iBody];
            riz = devDataSoA.qz[iBody];
        }

        for (int base_idx = 0; base_idx < n_bodies; base_idx += TILE_SIZE) {

            for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
                int gg_idx = base_idx + i;
                if (gg_idx < n_bodies) {
                    SHmg[i] = devGM[gg_idx];
                    SHqx[i] = devDataSoA.qx[gg_idx];
                    SHqy[i] = devDataSoA.qy[gg_idx];
                    SHqz[i] = devDataSoA.qz[gg_idx];
                } else {
                    SHmg[i] = T(0);
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

                    const T rijSquared =
                        rijx*rijx + rijy*rijy + rijz*rijz;

                    const T inv =
                        device_rsqrt<T>(rijSquared + softSquared);

                    potential_energy -=
                        iBodyMass * SHmg[jBody] * inv;
                }
            }
            __syncthreads();
        }

        if (iBody < n_bodies) {
            const T vx = devDataSoA.vx[iBody];
            const T vy = devDataSoA.vy[iBody];
            const T vz = devDataSoA.vz[iBody];

            kinetik_energy =
                iBodyMass * (vx*vx + vy*vy + vz*vz);
            potential_energy += iBodyMass * devGM[iBody] * device_rsqrt<T>(softSquared);
            bufferForEnergy[iBody] =
                potential_energy / Q(2)
              + kinetik_energy   / Q(2);

        }
    }
}



template <typename T, typename Q>
const accSoA_t<T>& SimulationNBodyCUDALeapfrog<T,Q>::getAccSoA() {
    accSoA.ax.resize(this->getBodies()->getN());
    accSoA.ay.resize(this->getBodies()->getN());
    accSoA.az.resize(this->getBodies()->getN());
    
    CUDA_CHECK(cudaMemcpy(accSoA.ax.data(), this->devAccelerations.x, this->getBodies()->getN() * sizeof(T), cudaMemcpyDeviceToHost));        
    CUDA_CHECK(cudaMemcpy(accSoA.ay.data(), this->devAccelerations.y, this->getBodies()->getN() * sizeof(T), cudaMemcpyDeviceToHost));     
    CUDA_CHECK(cudaMemcpy(accSoA.az.data(), this->devAccelerations.z, this->getBodies()->getN() * sizeof(T), cudaMemcpyDeviceToHost));        

    return accSoA;
}

template <typename T, typename Q>
void SimulationNBodyCUDALeapfrog<T,Q>::computeBodiesAcceleration()
{
    devComputeBodiesAccelerationLeapfrog<T><<<this->_num_blocks, this->_num_threads>>>(
                                            this->devAccelerations,
                                            this->cudaBodiesPtr->getDevDataSoA(),
                                            this->cudaBodiesPtr->getDevPositionsBuffer(),
                                            this->devGM,
                                            this->bodies->getN(), this->softSquared,
                                            this->_elem_per_thread);
    
    CUDA_CHECK(cudaGetLastError());
}

template <typename T, typename Q>
void SimulationNBodyCUDALeapfrog<T,Q>::computeMetrics() {
    devComputeBodiesMetrics<T,Q><<<this->_num_blocks, this->_num_threads>>>(
                            this->cudaBodiesPtr->getDevDataSoA(),
                            this->devGM,
                            this->bodies->getN(), this->softSquared,
                            this->_elem_per_thread,
                            this->bufferForEnergy);
    #if defined(COMPUTE_ALL_METRICS) || defined(COMPUTE_ENERGY_METRIC)
        void* d_temp = nullptr;
        size_t temp_bytes = 0;

        int numItems = this->bodies->getN();

        // query
        cub::DeviceReduce::Sum(
            d_temp, temp_bytes,
            bufferForEnergy,
            &(this->history->getDevEnergy()[this->currentIteration]),
            numItems
        );

        // alloc
        cudaMalloc(&d_temp, temp_bytes);

        // run
        cub::DeviceReduce::Sum(
            d_temp, temp_bytes,
            bufferForEnergy,
            &(this->history->getDevEnergy()[this->currentIteration]),
            numItems
        );

        cudaFree(d_temp);

    #endif
}

template <typename T, typename Q>
SimulationNBodyCUDALeapfrog<T,Q>::~SimulationNBodyCUDALeapfrog() {
    CUDA_CHECK(cudaFree(devAccelerations.x));
    CUDA_CHECK(cudaFree(devAccelerations.y));
    CUDA_CHECK(cudaFree(devAccelerations.z));
    CUDA_CHECK(cudaFree(this->bufferForEnergy));
}

template class SimulationNBodyCUDALeapfrog<float, double>;
template class SimulationNBodyCUDALeapfrog<float, float>;
template class SimulationNBodyCUDALeapfrog<double, double>;
// template class SimulationNBodyCUDALeapfrog<double>;

#endif