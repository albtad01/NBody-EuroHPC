#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <type_traits>

#include "mipp.h"
#include "SimulationNBodyOpenMP.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef MURB_OMP_RSQRT_NR
#define MURB_OMP_RSQRT_NR 0
#endif

#ifndef MURB_OMP_UNROLL
#define MURB_OMP_UNROLL 2
#endif

#ifndef MURB_OMP_PREFETCH_AHEAD
#define MURB_OMP_PREFETCH_AHEAD 4
#endif

template <typename T>
static inline T inv_sqrt_scalar(T x) {
    return T(1) / std::sqrt(x);
}

template <typename T>
static inline mipp::Reg<T> rsqrt_fast(const mipp::Reg<T>& x) {
    mipp::Reg<T> y = mipp::rsqrt(x);

    if constexpr (std::is_same<T, float>::value) {
#if MURB_OMP_RSQRT_NR >= 1
        const mipp::Reg<T> half = T(0.5f);
        const mipp::Reg<T> threehalfs = T(1.5f);
        y = y * (threehalfs - half * x * y * y);
#endif
#if MURB_OMP_RSQRT_NR >= 2
        const mipp::Reg<T> half = T(0.5f);
        const mipp::Reg<T> threehalfs = T(1.5f);
        y = y * (threehalfs - half * x * y * y);
#endif
    }
    return y;
}

template <typename T>
SimulationNBodyOpenMP<T>::SimulationNBodyOpenMP(const BodiesAllocatorInterface<T>& allocator, const T soft)
    : SimulationNBodyInterface<T>(allocator, soft)
{
    this->flopsPerIte = 20.f * (T)this->getBodies()->getN() * (T)this->getBodies()->getN();
    this->accelerations.resize(this->getBodies()->getN());
}

template <typename T>
void SimulationNBodyOpenMP<T>::initIteration()
{
}

template <typename T>
const std::vector<accAoS_t<T>>& SimulationNBodyOpenMP<T>::getAccAoS() {
    return accelerations;
}

template <typename T>
void SimulationNBodyOpenMP<T>::computeBodiesAcceleration()
{
    const auto& d = this->getBodies()->getDataSoA();
    const T* __restrict__ m  = d.m.data();
    const T* __restrict__ qx = d.qx.data();
    const T* __restrict__ qy = d.qy.data();
    const T* __restrict__ qz = d.qz.data();

    const long n = (long)this->getBodies()->getN();
    const T softSquared = this->soft * this->soft;
    const T G = this->G;

    auto* __restrict__ acc = this->accelerations.data();

    const int  V = mipp::N<T>();
    const long n_vec = n - (n % V);

#ifndef MURB_OMP_UNROLL
#define MURB_OMP_UNROLL 2
#endif
#ifndef MURB_OMP_PREFETCH_AHEAD
#define MURB_OMP_PREFETCH_AHEAD 4
#endif

#if (MURB_OMP_UNROLL != 1) && (MURB_OMP_UNROLL != 2) && (MURB_OMP_UNROLL != 4)
#error "MURB_OMP_UNROLL must be 1, 2, or 4"
#endif

    constexpr int UNROLL = MURB_OMP_UNROLL;
    const long step   = (long)UNROLL * (long)V;
    const long n_vec_u = n_vec - (n_vec % step);

#ifdef _OPENMP
    omp_set_dynamic(0);
#pragma omp parallel
#endif
    {
        const mipp::Reg<T> r_soft = softSquared;
        const mipp::Reg<T> r_G    = G;

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (long i = 0; i < n; i++) {

            const T qi_x = qx[i];
            const T qi_y = qy[i];
            const T qi_z = qz[i];

            const mipp::Reg<T> r_qi_x = qi_x;
            const mipp::Reg<T> r_qi_y = qi_y;
            const mipp::Reg<T> r_qi_z = qi_z;

            std::array<mipp::Reg<T>, UNROLL> r_ax;
            std::array<mipp::Reg<T>, UNROLL> r_ay;
            std::array<mipp::Reg<T>, UNROLL> r_az;

#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                r_ax[u] = T(0);
                r_ay[u] = T(0);
                r_az[u] = T(0);
            }

            long j = 0;

            for (; j < n_vec_u; j += step) {
                const long pf = j + (long)MURB_OMP_PREFETCH_AHEAD * (long)V;
                __builtin_prefetch(qx + pf, 0, 1);
                __builtin_prefetch(qy + pf, 0, 1);
                __builtin_prefetch(qz + pf, 0, 1);
                __builtin_prefetch(m  + pf, 0, 1);

#pragma unroll
                for (int u = 0; u < UNROLL; ++u) {
                    const long jj = j + (long)u * (long)V;

                    const mipp::Reg<T> r_mj  = &m[jj];
                    const mipp::Reg<T> r_qjx = &qx[jj];
                    const mipp::Reg<T> r_qjy = &qy[jj];
                    const mipp::Reg<T> r_qjz = &qz[jj];

                    const mipp::Reg<T> r_dx = r_qjx - r_qi_x;
                    const mipp::Reg<T> r_dy = r_qjy - r_qi_y;
                    const mipp::Reg<T> r_dz = r_qjz - r_qi_z;

                    const mipp::Reg<T> r_dist2 =
                        mipp::fmadd(r_dx, r_dx,
                        mipp::fmadd(r_dy, r_dy,
                        mipp::fmadd(r_dz, r_dz, r_soft)));

                    const mipp::Reg<T> r_inv  = rsqrt_fast<T>(r_dist2);
                    const mipp::Reg<T> r_inv2 = r_inv * r_inv;
                    const mipp::Reg<T> r_inv3 = r_inv2 * r_inv;

                    const mipp::Reg<T> r_fac = r_G * r_mj * r_inv3;

                    r_ax[u] = mipp::fmadd(r_fac, r_dx, r_ax[u]);
                    r_ay[u] = mipp::fmadd(r_fac, r_dy, r_ay[u]);
                    r_az[u] = mipp::fmadd(r_fac, r_dz, r_az[u]);
                }
            }

            for (; j < n_vec; j += V) {
                const mipp::Reg<T> r_mj  = &m[j];
                const mipp::Reg<T> r_qjx = &qx[j];
                const mipp::Reg<T> r_qjy = &qy[j];
                const mipp::Reg<T> r_qjz = &qz[j];

                const mipp::Reg<T> r_dx = r_qjx - r_qi_x;
                const mipp::Reg<T> r_dy = r_qjy - r_qi_y;
                const mipp::Reg<T> r_dz = r_qjz - r_qi_z;

                const mipp::Reg<T> r_dist2 =
                    mipp::fmadd(r_dx, r_dx,
                    mipp::fmadd(r_dy, r_dy,
                    mipp::fmadd(r_dz, r_dz, r_soft)));

                const mipp::Reg<T> r_inv  = rsqrt_fast<T>(r_dist2);
                const mipp::Reg<T> r_inv2 = r_inv * r_inv;
                const mipp::Reg<T> r_inv3 = r_inv2 * r_inv;

                const mipp::Reg<T> r_fac = r_G * r_mj * r_inv3;

                r_ax[0] = mipp::fmadd(r_fac, r_dx, r_ax[0]);
                r_ay[0] = mipp::fmadd(r_fac, r_dy, r_ay[0]);
                r_az[0] = mipp::fmadd(r_fac, r_dz, r_az[0]);
            }

            mipp::Reg<T> sum_ax = r_ax[0];
            mipp::Reg<T> sum_ay = r_ay[0];
            mipp::Reg<T> sum_az = r_az[0];

#pragma unroll
            for (int u = 1; u < UNROLL; ++u) {
                sum_ax += r_ax[u];
                sum_ay += r_ay[u];
                sum_az += r_az[u];
            }

            T ax = mipp::hadd(sum_ax);
            T ay = mipp::hadd(sum_ay);
            T az = mipp::hadd(sum_az);

            for (long jt = n_vec; jt < n; jt++) {
                const T dx = qx[jt] - qi_x;
                const T dy = qy[jt] - qi_y;
                const T dz = qz[jt] - qi_z;

                const T dist2 = dx*dx + dy*dy + dz*dz + softSquared;
                const T inv   = inv_sqrt_scalar<T>(dist2);
                const T inv3  = (inv * inv) * inv;
                const T fac   = G * m[jt] * inv3;

                ax += fac * dx;
                ay += fac * dy;
                az += fac * dz;
            }

            acc[i].ax = ax;
            acc[i].ay = ay;
            acc[i].az = az;
        }
    }
}


template <typename T>
void SimulationNBodyOpenMP<T>::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    this->bodies->updatePositionsAndVelocities(this->accelerations, this->dt);
}

template class SimulationNBodyOpenMP<float>;
template class SimulationNBodyOpenMP<double>;
