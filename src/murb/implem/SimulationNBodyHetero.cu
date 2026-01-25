#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <type_traits>
#include <cstdio>

#include "SimulationNBodyHetero.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cuda_runtime.h>

#ifndef MURB_HETERO_GPU_FRACTION
#define MURB_HETERO_GPU_FRACTION 0.60
#endif

#ifndef MURB_HETERO_BLOCK
#define MURB_HETERO_BLOCK 256
#endif

#ifndef MURB_HETERO_MIN_N
#define MURB_HETERO_MIN_N 8192
#endif

static inline void cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
        std::abort();
    }
}

__device__ __forceinline__ float inv_sqrt_device(float x) {
    return 1.0f / sqrtf(x);
}

__device__ __forceinline__ double inv_sqrt_device(double x) {
    return 1.0 / sqrt(x);
}

template <typename T>
__global__ void nbody_acc_kernel(
    const T* __restrict__ m,
    const T* __restrict__ qx,
    const T* __restrict__ qy,
    const T* __restrict__ qz,
    T* __restrict__ ax,
    T* __restrict__ ay,
    T* __restrict__ az,
    long n,
    long i0,
    long i1,
    T soft2,
    T G)
{
    const long tid = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
    const long i = i0 + tid;
    if (i >= i1) return;

    const T qi_x = qx[i];
    const T qi_y = qy[i];
    const T qi_z = qz[i];

    T accx = (T)0;
    T accy = (T)0;
    T accz = (T)0;

    __shared__ T sh_m[MURB_HETERO_BLOCK];
    __shared__ T sh_x[MURB_HETERO_BLOCK];
    __shared__ T sh_y[MURB_HETERO_BLOCK];
    __shared__ T sh_z[MURB_HETERO_BLOCK];

    for (long base = 0; base < n; base += (long)blockDim.x) {
        const long j = base + (long)threadIdx.x;
        if (j < n) {
            sh_m[threadIdx.x] = m[j];
            sh_x[threadIdx.x] = qx[j];
            sh_y[threadIdx.x] = qy[j];
            sh_z[threadIdx.x] = qz[j];
        } else {
            sh_m[threadIdx.x] = (T)0;
            sh_x[threadIdx.x] = (T)0;
            sh_y[threadIdx.x] = (T)0;
            sh_z[threadIdx.x] = (T)0;
        }
        __syncthreads();

        const long rem = n - base;
        const long lim = rem < (long)blockDim.x ? rem : (long)blockDim.x;

        #pragma unroll 4
        for (long t = 0; t < lim; ++t) {
            const T dx = sh_x[t] - qi_x;
            const T dy = sh_y[t] - qi_y;
            const T dz = sh_z[t] - qi_z;

            const T dist2 = dx*dx + dy*dy + dz*dz + soft2;

            const T inv  = inv_sqrt_device((typename std::conditional<std::is_same<T,float>::value,float,double>::type)dist2);
            const T inv3 = (inv * inv) * inv;

            const T fac = G * sh_m[t] * inv3;

            accx += fac * dx;
            accy += fac * dy;
            accz += fac * dz;
        }
        __syncthreads();
    }

    ax[i - i0] = accx;
    ay[i - i0] = accy;
    az[i - i0] = accz;
}

template <typename T>
SimulationNBodyHetero<T>::SimulationNBodyHetero(const BodiesAllocatorInterface<T>& allocator, const T soft)
: SimulationNBodyInterface<T>(allocator, soft)
{
    this->flopsPerIte = 20.f * (T)this->getBodies()->getN() * (T)this->getBodies()->getN();
    accelerations.resize(this->getBodies()->getN());
}

template <typename T>
SimulationNBodyHetero<T>::~SimulationNBodyHetero()
{
    releaseCuda();
}

template <typename T>
const std::vector<accAoS_t<T>>& SimulationNBodyHetero<T>::getAccAoS() {
    return accelerations;
}

template <typename T>
void SimulationNBodyHetero<T>::releaseCuda()
{
    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
    if (d_m)  { cudaFree(d_m);  d_m  = nullptr; }
    if (d_qx) { cudaFree(d_qx); d_qx = nullptr; }
    if (d_qy) { cudaFree(d_qy); d_qy = nullptr; }
    if (d_qz) { cudaFree(d_qz); d_qz = nullptr; }
    if (d_ax) { cudaFree(d_ax); d_ax = nullptr; }
    if (d_ay) { cudaFree(d_ay); d_ay = nullptr; }
    if (d_az) { cudaFree(d_az); d_az = nullptr; }
    cap_n = 0;
    cap_cut = 0;
    hax.clear();
    hay.clear();
    haz.clear();
}

template <typename T>
void SimulationNBodyHetero<T>::ensureCuda(std::size_t n, std::size_t cut)
{
    if (!stream) {
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    }

    if (n > cap_n) {
        if (d_m)  cudaFree(d_m);
        if (d_qx) cudaFree(d_qx);
        if (d_qy) cudaFree(d_qy);
        if (d_qz) cudaFree(d_qz);

        cuda_check(cudaMalloc((void**)&d_m,  n * sizeof(T)), "cudaMalloc d_m");
        cuda_check(cudaMalloc((void**)&d_qx, n * sizeof(T)), "cudaMalloc d_qx");
        cuda_check(cudaMalloc((void**)&d_qy, n * sizeof(T)), "cudaMalloc d_qy");
        cuda_check(cudaMalloc((void**)&d_qz, n * sizeof(T)), "cudaMalloc d_qz");

        cap_n = n;
    }

    if (cut > cap_cut) {
        if (d_ax) cudaFree(d_ax);
        if (d_ay) cudaFree(d_ay);
        if (d_az) cudaFree(d_az);

        cuda_check(cudaMalloc((void**)&d_ax, cut * sizeof(T)), "cudaMalloc d_ax");
        cuda_check(cudaMalloc((void**)&d_ay, cut * sizeof(T)), "cudaMalloc d_ay");
        cuda_check(cudaMalloc((void**)&d_az, cut * sizeof(T)), "cudaMalloc d_az");

        cap_cut = cut;
        hax.resize(cut);
        hay.resize(cut);
        haz.resize(cut);
    } else {
        if (hax.size() != cut) {
            hax.resize(cut);
            hay.resize(cut);
            haz.resize(cut);
        }
    }
}

template <typename T>
void SimulationNBodyHetero<T>::initIteration() {}

template <typename T>
void SimulationNBodyHetero<T>::computeBodiesAcceleration()
{
    const auto& d = this->getBodies()->getDataSoA();
    const T* __restrict__ m  = d.m.data();
    const T* __restrict__ qx = d.qx.data();
    const T* __restrict__ qy = d.qy.data();
    const T* __restrict__ qz = d.qz.data();

    const long n = (long)this->getBodies()->getN();
    const T soft2 = this->soft * this->soft;
    const T G = this->G;

    long min_n = (long)MURB_HETERO_MIN_N;
    if (const char* s = std::getenv("MURB_HETERO_MIN_N")) {
        min_n = std::max(0L, std::atol(s));
    }

    double frac = MURB_HETERO_GPU_FRACTION;
    if (const char* s = std::getenv("MURB_HETERO_GPU_FRACTION")) {
        frac = std::max(0.0, std::min(1.0, std::atof(s)));
    }
    long cut = (long)(frac * (double)n);
    cut = std::max(0L, std::min(n, cut));

    int devCount = 0;
    cudaError_t ce = cudaGetDeviceCount(&devCount);

    const bool use_gpu = (n >= min_n) && (ce == cudaSuccess) && (devCount > 0) && (cut > 0);

    if (!use_gpu) {
#ifdef _OPENMP
        omp_set_dynamic(0);
#pragma omp parallel for schedule(static)
#endif
        for (long i = 0; i < n; i++) {
            const T qi_x = qx[i], qi_y = qy[i], qi_z = qz[i];
            T ax = (T)0, ay = (T)0, az = (T)0;
            for (long j = 0; j < n; j++) {
                const T dx = qx[j] - qi_x;
                const T dy = qy[j] - qi_y;
                const T dz = qz[j] - qi_z;
                const T dist2 = dx*dx + dy*dy + dz*dz + soft2;
                const T inv = (T)1 / std::sqrt(dist2);
                const T inv3 = (inv*inv)*inv;
                const T fac = G * m[j] * inv3;
                ax += fac * dx; ay += fac * dy; az += fac * dz;
            }
            accelerations[i].ax = ax;
            accelerations[i].ay = ay;
            accelerations[i].az = az;
        }
        return;
    }

    ensureCuda((std::size_t)n, (std::size_t)cut);

    cuda_check(cudaMemcpyAsync(d_m,  m,  (std::size_t)n * sizeof(T), cudaMemcpyHostToDevice, stream), "H2D m");
    cuda_check(cudaMemcpyAsync(d_qx, qx, (std::size_t)n * sizeof(T), cudaMemcpyHostToDevice, stream), "H2D qx");
    cuda_check(cudaMemcpyAsync(d_qy, qy, (std::size_t)n * sizeof(T), cudaMemcpyHostToDevice, stream), "H2D qy");
    cuda_check(cudaMemcpyAsync(d_qz, qz, (std::size_t)n * sizeof(T), cudaMemcpyHostToDevice, stream), "H2D qz");

    const int block = MURB_HETERO_BLOCK;
    const int grid  = (int)(((std::size_t)cut + (std::size_t)block - 1) / (std::size_t)block);

    nbody_acc_kernel<T><<<grid, block, 0, stream>>>(
        d_m, d_qx, d_qy, d_qz, d_ax, d_ay, d_az, n, 0, cut, soft2, G
    );
    cuda_check(cudaGetLastError(), "kernel launch");

#ifdef _OPENMP
    omp_set_dynamic(0);
#pragma omp parallel for schedule(static)
#endif
    for (long i = cut; i < n; i++) {
        const T qi_x = qx[i], qi_y = qy[i], qi_z = qz[i];
        T ax = (T)0, ay = (T)0, az = (T)0;
        for (long j = 0; j < n; j++) {
            const T dx = qx[j] - qi_x;
            const T dy = qy[j] - qi_y;
            const T dz = qz[j] - qi_z;
            const T dist2 = dx*dx + dy*dy + dz*dz + soft2;
            const T inv = (T)1 / std::sqrt(dist2);
            const T inv3 = (inv*inv)*inv;
            const T fac = G * m[j] * inv3;
            ax += fac * dx; ay += fac * dy; az += fac * dz;
        }
        accelerations[i].ax = ax;
        accelerations[i].ay = ay;
        accelerations[i].az = az;
    }

    cuda_check(cudaMemcpyAsync(hax.data(), d_ax, (std::size_t)cut * sizeof(T), cudaMemcpyDeviceToHost, stream), "D2H ax");
    cuda_check(cudaMemcpyAsync(hay.data(), d_ay, (std::size_t)cut * sizeof(T), cudaMemcpyDeviceToHost, stream), "D2H ay");
    cuda_check(cudaMemcpyAsync(haz.data(), d_az, (std::size_t)cut * sizeof(T), cudaMemcpyDeviceToHost, stream), "D2H az");

    cuda_check(cudaStreamSynchronize(stream), "stream sync");

    for (long i = 0; i < cut; i++) {
        accelerations[i].ax = hax[(std::size_t)i];
        accelerations[i].ay = hay[(std::size_t)i];
        accelerations[i].az = haz[(std::size_t)i];
    }
}

template <typename T>
void SimulationNBodyHetero<T>::computeOneIteration()
{
    initIteration();
    computeBodiesAcceleration();
    this->bodies->updatePositionsAndVelocities(accelerations, this->dt);
}

template class SimulationNBodyHetero<float>;
template class SimulationNBodyHetero<double>;
