#ifndef SIMULATION_N_BODY_CUDA_TILE_FULL_DEVICE_CU_
#define SIMULATION_N_BODY_CUDA_TILE_FULL_DEVICE_CU_

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include "SimulationNBodyCUDATileFullDevice.hpp"

#define CUDA_CHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char *file, int line) {
    if (error_code != cudaSuccess) {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n",
                (int)error_code, cudaGetErrorString(error_code), file, line);
        std::exit((int)error_code);
    }
}

template <typename T>
__device__ __forceinline__ T device_rsqrt(T val);

template <> __device__ __forceinline__ float device_rsqrt<float>(float val) {
    // rsqrtf + --use_fast_math => molto veloce
    return rsqrtf(val);
}
template <> __device__ __forceinline__ double device_rsqrt<double>(double val) {
    // double sarà molto più lento: su A100 punta a float se puoi
    return rsqrt(val);
}

template <typename T>
__device__ __forceinline__ T fmadd(T a, T b, T c) { return a*b + c; }

template <>
__device__ __forceinline__ float fmadd<float>(float a, float b, float c) { return fmaf(a,b,c); }

template <>
__device__ __forceinline__ double fmadd<double>(double a, double b, double c) { return fma(a,b,c); }

// Efficient initialization of G * Mass (GM)
template <typename T>
__global__ void devInitializeDevGM(const devDataSoA_t<T> devDataSoA, const int n_bodies, T G, T* __restrict__ devGM) {
    const int iBody = blockIdx.x * blockDim.x + threadIdx.x;
    if (iBody < n_bodies) devGM[iBody] = G * devDataSoA.m[iBody];
}

/*
  Full-device N^2 kernel with tiling.
  BLOCK is fixed to 256.
  EPT can be 2 (TILE=512) or 4 (TILE=1024).
  A100 often benefits from EPT=2 (lower reg pressure -> higher occupancy).
*/
template <typename T, int BLOCK, int EPT>
__global__ __launch_bounds__(BLOCK, 2)
void devComputeBodiesAccelerationTileFullDevice_opt(
    devAccSoA_t<T> devAcc,
    const devDataSoA_t<T> devData,
    const T* __restrict__ devGM,
    const int n_bodies,
    const T softSquared
) {
    static_assert(BLOCK == 256, "This kernel is tuned for BLOCK=256");
    static_assert(EPT == 2 || EPT == 4, "EPT must be 2 or 4");

    constexpr int TILE = BLOCK * EPT;

    __shared__ T SHm[TILE];
    __shared__ T SHqx[TILE];
    __shared__ T SHqy[TILE];
    __shared__ T SHqz[TILE];

    const T* __restrict__ qx = devData.qx;
    const T* __restrict__ qy = devData.qy;
    const T* __restrict__ qz = devData.qz;

    T accX[EPT], accY[EPT], accZ[EPT];
    T rix[EPT],  riy[EPT],  riz[EPT];

    #pragma unroll
    for (int k = 0; k < EPT; ++k) {
        accX[k] = accY[k] = accZ[k] = T(0);
        int iBody = blockIdx.x * (BLOCK * EPT) + threadIdx.x + k * BLOCK;
        if (iBody < n_bodies) {
            rix[k] = qx[iBody];
            riy[k] = qy[iBody];
            riz[k] = qz[iBody];
        } else {
            // evita NaN se fuori range
            rix[k] = riy[k] = riz[k] = T(0);
        }
    }

    for (int base = 0; base < n_bodies; base += TILE) {

        // Load tile to shared (coalesced)
        for (int t = threadIdx.x; t < TILE; t += BLOCK) {
            int j = base + t;
            if (j < n_bodies) {
                SHm[t]  = devGM[j];
                SHqx[t] = qx[j];
                SHqy[t] = qy[j];
                SHqz[t] = qz[j];
            } else {
                SHm[t] = SHqx[t] = SHqy[t] = SHqz[t] = T(0);
            }
        }
        __syncthreads();

        // Accumulate interactions within this tile
        // Unroll moderato per non far esplodere i registri
        #pragma unroll 4
        for (int j = 0; j < TILE; ++j) {
            const T sjx = SHqx[j];
            const T sjy = SHqy[j];
            const T sjz = SHqz[j];
            const T sm  = SHm[j];

            #pragma unroll
            for (int k = 0; k < EPT; ++k) {
                const T rijx = sjx - rix[k];
                const T rijy = sjy - riy[k];
                const T rijz = sjz - riz[k];

                // distSq = rijx^2 + rijy^2 + rijz^2 + soft^2 (FMA)
                T distSq = fmadd(rijx, rijx, softSquared);
                distSq   = fmadd(rijy, rijy, distSq);
                distSq   = fmadd(rijz, rijz, distSq);

                const T invDist  = device_rsqrt<T>(distSq);
                const T invDist2 = invDist * invDist;
                const T invDist3 = invDist2 * invDist;

                const T f = sm * invDist3;

                accX[k] = fmadd(f, rijx, accX[k]);
                accY[k] = fmadd(f, rijy, accY[k]);
                accZ[k] = fmadd(f, rijz, accZ[k]);
            }
        }
        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int k = 0; k < EPT; ++k) {
        int iBody = blockIdx.x * (BLOCK * EPT) + threadIdx.x + k * BLOCK;
        if (iBody < n_bodies) {
            devAcc.x[iBody] = accX[k];
            devAcc.y[iBody] = accY[k];
            devAcc.z[iBody] = accZ[k];
        }
    }
}

template <typename T>
SimulationNBodyCUDATileFullDevice<T>::SimulationNBodyCUDATileFullDevice(
    const BodiesAllocatorInterface<T>& allocator,
    const T soft,
    const bool transfer_each_iteration
)
: SimulationNBodyInterface<T>(allocator, soft),
  softSquared{soft*soft},
  transfer_each_iteration{transfer_each_iteration}
{
    const int n = (int)this->getBodies()->getN();
    this->flopsPerIte = 20.f * (T)n * (T)n;

    // Detect GPU and choose EPT (A100 tends to like EPT=2 for occupancy)
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    // Default tuning
    this->_num_threads = 256;

    // SM80 = A100
    // const bool is_a100 = (prop.major == 8 && prop.minor == 0);
    // this->_elem_per_thread = is_a100 ? 2 : 4;
    this->_elem_per_thread = 4;


    const int ept = this->_elem_per_thread;
    this->_num_blocks = (n + (this->_num_threads * ept) - 1) / (this->_num_threads * ept);

    CUDA_CHECK(cudaMalloc(&this->devAccelerations.x, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&this->devAccelerations.y, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&this->devAccelerations.z, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&this->devGM, n * sizeof(T)));

    this->cudaBodiesPtr = std::dynamic_pointer_cast<CUDABodies<T>>(this->bodies);

    // Init GM
    const int init_blocks = (n + 255) / 256;
    devInitializeDevGM<T><<<init_blocks, 256>>>(
        this->cudaBodiesPtr->getDevDataSoA(), n, this->G, this->devGM
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename T>
void SimulationNBodyCUDATileFullDevice<T>::computeOneIteration() {
    const int n = (int)this->bodies->getN();

    if (this->_elem_per_thread == 2) {
        devComputeBodiesAccelerationTileFullDevice_opt<T, 256, 2>
            <<<this->_num_blocks, this->_num_threads>>>(
                this->devAccelerations,
                this->cudaBodiesPtr->getDevDataSoA(),
                this->devGM,
                n,
                this->softSquared
            );
    } else {
        devComputeBodiesAccelerationTileFullDevice_opt<T, 256, 4>
            <<<this->_num_blocks, this->_num_threads>>>(
                this->devAccelerations,
                this->cudaBodiesPtr->getDevDataSoA(),
                this->devGM,
                n,
                this->softSquared
            );
    }

    CUDA_CHECK(cudaGetLastError());

    // IMPORTANT: qui conta moltissimo se questa funzione fa sync o transfer.
    // Assicurati che sia device-only e NON faccia copie host<->device a ogni iterazione.
    this->cudaBodiesPtr->updatePositionsAndVelocitiesOnDevice(this->devAccelerations, this->dt);

    if (this->transfer_each_iteration) {
        // Questo può uccidere le performance (host transfer!)
        this->cudaBodiesPtr->getDataSoA();
    }
}

template <typename T>
SimulationNBodyCUDATileFullDevice<T>::~SimulationNBodyCUDATileFullDevice() {
    cudaFree(devAccelerations.x);
    cudaFree(devAccelerations.y);
    cudaFree(devAccelerations.z);
    cudaFree(devGM);
}

template class SimulationNBodyCUDATileFullDevice<float>;
template class SimulationNBodyCUDATileFullDevice<double>;

#endif
