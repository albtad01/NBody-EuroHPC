#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <vector>

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

// -------------------- CUDA helpers --------------------
static inline void cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
        std::abort();
    }
}

// Kernel: each thread computes one i in [i0, i1)
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

    // Tiling over j
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

        const long lim = min((long)blockDim.x, n - base);
        #pragma unroll 4
        for (long t = 0; t < lim; ++t) {
            const T dx = sh_x[t] - qi_x;
            const T dy = sh_y[t] - qi_y;
            const T dz = sh_z[t] - qi_z;

            const T dist2 = dx*dx + dy*dy + dz*dz + soft2;

            T inv;
            if constexpr (std::is_same<T, float>::value) {
                inv = rsqrtf((float)dist2);
            } else {
                inv = rsqrt((double)dist2);
            }
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

// -------------------- Class impl --------------------
template <typename T>
SimulationNBodyHetero<T>::SimulationNBodyHetero(const BodiesAllocatorInterface<T>& allocator, const T soft)
: SimulationNBodyInterface<T>(allocator, soft)
{
    this->flopsPerIte = 20.f * (T)this->getBodies()->getN() * (T)this->getBodies()->getN();
    accelerations.resize(this->getBodies()->getN());
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

    // Decide split ratio (env overrides)
    double frac = MURB_HETERO_GPU_FRACTION;
    if (const char* s = std::getenv("MURB_HETERO_GPU_FRACTION")) {
        frac = std::max(0.0, std::min(1.0, std::atof(s)));
    }
    long cut = (long)(frac * (double)n);
    cut = std::max(0L, std::min(n, cut));

    // If no GPU work or no CUDA device, fallback CPU-only
    int devCount = 0;
    cudaError_t ce = cudaGetDeviceCount(&devCount);
    if (ce != cudaSuccess || devCount <= 0 || cut == 0) {
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
            this->accelerations[i].ax = ax;
            this->accelerations[i].ay = ay;
            this->accelerations[i].az = az;
        }
        return;
    }

    // Allocate device buffers
    T *d_m=nullptr, *d_qx=nullptr, *d_qy=nullptr, *d_qz=nullptr;
    T *d_ax=nullptr, *d_ay=nullptr, *d_az=nullptr;

    cuda_check(cudaMalloc((void**)&d_m,  (size_t)n * sizeof(T)), "cudaMalloc d_m");
    cuda_check(cudaMalloc((void**)&d_qx, (size_t)n * sizeof(T)), "cudaMalloc d_qx");
    cuda_check(cudaMalloc((void**)&d_qy, (size_t)n * sizeof(T)), "cudaMalloc d_qy");
    cuda_check(cudaMalloc((void**)&d_qz, (size_t)n * sizeof(T)), "cudaMalloc d_qz");

    cuda_check(cudaMalloc((void**)&d_ax, (size_t)cut * sizeof(T)), "cudaMalloc d_ax");
    cuda_check(cudaMalloc((void**)&d_ay, (size_t)cut * sizeof(T)), "cudaMalloc d_ay");
    cuda_check(cudaMalloc((void**)&d_az, (size_t)cut * sizeof(T)), "cudaMalloc d_az");

    cudaStream_t stream;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

    // Copy inputs
    cuda_check(cudaMemcpyAsync(d_m,  m,  (size_t)n * sizeof(T), cudaMemcpyHostToDevice, stream), "H2D m");
    cuda_check(cudaMemcpyAsync(d_qx, qx, (size_t)n * sizeof(T), cudaMemcpyHostToDevice, stream), "H2D qx");
    cuda_check(cudaMemcpyAsync(d_qy, qy, (size_t)n * sizeof(T), cudaMemcpyHostToDevice, stream), "H2D qy");
    cuda_check(cudaMemcpyAsync(d_qz, qz, (size_t)n * sizeof(T), cudaMemcpyHostToDevice, stream), "H2D qz");

    // Launch kernel for i in [0, cut)
    const int block = MURB_HETERO_BLOCK;
    const int grid  = (int)((cut + block - 1) / block);

    nbody_acc_kernel<T><<<grid, block, 0, stream>>>(
        d_m, d_qx, d_qy, d_qz, d_ax, d_ay, d_az, n, 0, cut, soft2, G
    );
    cuda_check(cudaGetLastError(), "kernel launch");

    // CPU computes [cut, n)
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
        this->accelerations[i].ax = ax;
        this->accelerations[i].ay = ay;
        this->accelerations[i].az = az;
    }

    // Copy GPU results back into accelerations[0..cut)
    std::vector<T> hax((size_t)cut), hay((size_t)cut), haz((size_t)cut);

    cuda_check(cudaMemcpyAsync(hax.data(), d_ax, (size_t)cut * sizeof(T), cudaMemcpyDeviceToHost, stream), "D2H ax");
    cuda_check(cudaMemcpyAsync(hay.data(), d_ay, (size_t)cut * sizeof(T), cudaMemcpyDeviceToHost, stream), "D2H ay");
    cuda_check(cudaMemcpyAsync(haz.data(), d_az, (size_t)cut * sizeof(T), cudaMemcpyDeviceToHost, stream), "D2H az");

    cuda_check(cudaStreamSynchronize(stream), "stream sync");

    for (long i = 0; i < cut; i++) {
        this->accelerations[i].ax = hax[(size_t)i];
        this->accelerations[i].ay = hay[(size_t)i];
        this->accelerations[i].az = haz[(size_t)i];
    }

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_m);  cudaFree(d_qx); cudaFree(d_qy); cudaFree(d_qz);
    cudaFree(d_ax); cudaFree(d_ay); cudaFree(d_az);
}

template <typename T>
void SimulationNBodyHetero<T>::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    this->bodies->updatePositionsAndVelocities(this->accelerations, this->dt);
}

// Explicit instantiation
template class SimulationNBodyHetero<float>;
template class SimulationNBodyHetero<double>;
