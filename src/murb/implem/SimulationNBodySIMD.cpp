#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>

#include "mipp.h"
#include "SimulationNBodySIMD.hpp"

#ifndef MURB_SIMD_RSQRT_NR
#define MURB_SIMD_RSQRT_NR 0
#endif

template <typename T>
static inline T inv_sqrt_scalar(T x) {
    return T(1) / std::sqrt(x);
}

template <typename T>
static inline mipp::Reg<T> rsqrt_fast(const mipp::Reg<T>& x) {
    mipp::Reg<T> y = mipp::rsqrt(x);
    if constexpr (std::is_same<T, float>::value) {
#if MURB_SIMD_RSQRT_NR >= 1
        const mipp::Reg<T> half = T(0.5f);
        const mipp::Reg<T> threehalfs = T(1.5f);
        y = y * (threehalfs - half * x * y * y);
#endif
    }
    return y;
}

template <typename T>
SimulationNBodySIMD<T>::SimulationNBodySIMD(const BodiesAllocatorInterface<T>& allocator, const T soft)
    : SimulationNBodyInterface<T>(allocator, soft)
{
    this->flopsPerIte = 20.f * (T)this->getBodies()->getN() * (T)this->getBodies()->getN();
    this->accelerations.resize(this->getBodies()->getN());
}

template <typename T>
void SimulationNBodySIMD<T>::initIteration()
{
}

template <typename T>
const std::vector<accAoS_t<T>>& SimulationNBodySIMD<T>::getAccAoS() {
    return accelerations;
}

template <typename T>
void SimulationNBodySIMD<T>::computeBodiesAcceleration()
{
    const auto& d = this->getBodies()->getDataSoA();
    const T* __restrict__ m  = d.m.data();
    const T* __restrict__ qx = d.qx.data();
    const T* __restrict__ qy = d.qy.data();
    const T* __restrict__ qz = d.qz.data();

    const long n = (long)this->getBodies()->getN();

    const T softSquared = this->soft * this->soft;
    const T G = this->G;

    const int V = mipp::N<T>();
    const long n_vec = n - (n % V);

    constexpr int UNROLL = 4;
    const long n_vec_u = n_vec - (n_vec % (UNROLL * V));

    for (long i = 0; i < n; i++) {

        const T qi_x = qx[i];
        const T qi_y = qy[i];
        const T qi_z = qz[i];

        const mipp::Reg<T> r_qi_x = qi_x;
        const mipp::Reg<T> r_qi_y = qi_y;
        const mipp::Reg<T> r_qi_z = qi_z;
        const mipp::Reg<T> r_soft = softSquared;
        const mipp::Reg<T> r_G    = G;

        mipp::Reg<T> r_ax0 = T(0), r_ay0 = T(0), r_az0 = T(0);
        mipp::Reg<T> r_ax1 = T(0), r_ay1 = T(0), r_az1 = T(0);
        mipp::Reg<T> r_ax2 = T(0), r_ay2 = T(0), r_az2 = T(0);
        mipp::Reg<T> r_ax3 = T(0), r_ay3 = T(0), r_az3 = T(0);

        long j = 0;

        for (; j < n_vec_u; j += UNROLL * V) {

            {
                const long jj = j + 0 * V;
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

                mipp::Reg<T> r_inv;
                if constexpr (std::is_same<T, float>::value) r_inv = rsqrt_fast<T>(r_dist2);
                else r_inv = mipp::rsqrt(r_dist2);

                const mipp::Reg<T> r_inv2 = r_inv * r_inv;
                const mipp::Reg<T> r_inv3 = r_inv2 * r_inv;

                const mipp::Reg<T> r_fac = r_G * r_mj * r_inv3;

                r_ax0 = mipp::fmadd(r_fac, r_dx, r_ax0);
                r_ay0 = mipp::fmadd(r_fac, r_dy, r_ay0);
                r_az0 = mipp::fmadd(r_fac, r_dz, r_az0);
            }

            {
                const long jj = j + 1 * V;
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

                mipp::Reg<T> r_inv;
                if constexpr (std::is_same<T, float>::value) r_inv = rsqrt_fast<T>(r_dist2);
                else r_inv = mipp::rsqrt(r_dist2);

                const mipp::Reg<T> r_inv2 = r_inv * r_inv;
                const mipp::Reg<T> r_inv3 = r_inv2 * r_inv;

                const mipp::Reg<T> r_fac = r_G * r_mj * r_inv3;

                r_ax1 = mipp::fmadd(r_fac, r_dx, r_ax1);
                r_ay1 = mipp::fmadd(r_fac, r_dy, r_ay1);
                r_az1 = mipp::fmadd(r_fac, r_dz, r_az1);
            }

            {
                const long jj = j + 2 * V;
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

                mipp::Reg<T> r_inv;
                if constexpr (std::is_same<T, float>::value) r_inv = rsqrt_fast<T>(r_dist2);
                else r_inv = mipp::rsqrt(r_dist2);

                const mipp::Reg<T> r_inv2 = r_inv * r_inv;
                const mipp::Reg<T> r_inv3 = r_inv2 * r_inv;

                const mipp::Reg<T> r_fac = r_G * r_mj * r_inv3;

                r_ax2 = mipp::fmadd(r_fac, r_dx, r_ax2);
                r_ay2 = mipp::fmadd(r_fac, r_dy, r_ay2);
                r_az2 = mipp::fmadd(r_fac, r_dz, r_az2);
            }

            {
                const long jj = j + 3 * V;
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

                mipp::Reg<T> r_inv;
                if constexpr (std::is_same<T, float>::value) r_inv = rsqrt_fast<T>(r_dist2);
                else r_inv = mipp::rsqrt(r_dist2);

                const mipp::Reg<T> r_inv2 = r_inv * r_inv;
                const mipp::Reg<T> r_inv3 = r_inv2 * r_inv;

                const mipp::Reg<T> r_fac = r_G * r_mj * r_inv3;

                r_ax3 = mipp::fmadd(r_fac, r_dx, r_ax3);
                r_ay3 = mipp::fmadd(r_fac, r_dy, r_ay3);
                r_az3 = mipp::fmadd(r_fac, r_dz, r_az3);
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

            mipp::Reg<T> r_inv;
            if constexpr (std::is_same<T, float>::value) r_inv = rsqrt_fast<T>(r_dist2);
            else r_inv = mipp::rsqrt(r_dist2);

            const mipp::Reg<T> r_inv2 = r_inv * r_inv;
            const mipp::Reg<T> r_inv3 = r_inv2 * r_inv;

            const mipp::Reg<T> r_fac = r_G * r_mj * r_inv3;

            r_ax0 = mipp::fmadd(r_fac, r_dx, r_ax0);
            r_ay0 = mipp::fmadd(r_fac, r_dy, r_ay0);
            r_az0 = mipp::fmadd(r_fac, r_dz, r_az0);
        }

        const mipp::Reg<T> r_ax = (r_ax0 + r_ax1) + (r_ax2 + r_ax3);
        const mipp::Reg<T> r_ay = (r_ay0 + r_ay1) + (r_ay2 + r_ay3);
        const mipp::Reg<T> r_az = (r_az0 + r_az1) + (r_az2 + r_az3);

        T ax = mipp::hadd(r_ax);
        T ay = mipp::hadd(r_ay);
        T az = mipp::hadd(r_az);

        for (long jt = n_vec; jt < n; jt++) {
            const T dx = qx[jt] - qi_x;
            const T dy = qy[jt] - qi_y;
            const T dz = qz[jt] - qi_z;

            const T dist2 = dx*dx + dy*dy + dz*dz + softSquared;

            T inv;
            if constexpr (std::is_same<T, float>::value) inv = inv_sqrt_scalar<T>(dist2);
            else inv = inv_sqrt_scalar<T>(dist2);

            const T inv3 = (inv * inv) * inv;

            const T fac = G * m[jt] * inv3;

            ax += fac * dx;
            ay += fac * dy;
            az += fac * dz;
        }

        this->accelerations[i].ax = ax;
        this->accelerations[i].ay = ay;
        this->accelerations[i].az = az;
    }
}

template <typename T>
void SimulationNBodySIMD<T>::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    this->bodies->updatePositionsAndVelocities(this->accelerations, this->dt);
}

template class SimulationNBodySIMD<float>;
template class SimulationNBodySIMD<double>;
