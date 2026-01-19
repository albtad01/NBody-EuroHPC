#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "mipp.h"
#include "SimulationNBodySIMD.hpp"

template <typename T>
static inline T inv_sqrt_scalar(T x) {
    // Portable scalar fallback. (No rsqrtf on CPU.)
    return T(1) / std::sqrt(x);
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
    std::fill(this->accelerations.begin(), this->accelerations.end(), accAoS_t<T>{T(0), T(0), T(0)});
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

    const T softSquared = this->soft * this->soft;  // hoist invariant
    const T G = this->G;                            // hoist invariant

    const int V = mipp::N<T>();
    const long n_vec = n - (n % V);

    constexpr int UNROLL = 2; // try 1/2/4 and keep the best

    for (long i = 0; i < n; i++) {

        const T qi_x = qx[i];
        const T qi_y = qy[i];
        const T qi_z = qz[i];

        const mipp::Reg<T> r_qi_x = qi_x;
        const mipp::Reg<T> r_qi_y = qi_y;
        const mipp::Reg<T> r_qi_z = qi_z;
        const mipp::Reg<T> r_soft = softSquared;
        const mipp::Reg<T> r_G    = G;

        mipp::Reg<T> r_ax = T(0);
        mipp::Reg<T> r_ay = T(0);
        mipp::Reg<T> r_az = T(0);

        long j = 0;

        // Unrolled vector loop
        for (; j + (UNROLL * V) <= n_vec; j += UNROLL * V) {
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const long jj = j + u * V;

                const mipp::Reg<T> r_mj  = &m[jj];
                const mipp::Reg<T> r_qjx = &qx[jj];
                const mipp::Reg<T> r_qjy = &qy[jj];
                const mipp::Reg<T> r_qjz = &qz[jj];

                const mipp::Reg<T> r_dx = r_qjx - r_qi_x;
                const mipp::Reg<T> r_dy = r_qjy - r_qi_y;
                const mipp::Reg<T> r_dz = r_qjz - r_qi_z;

                // Use FMA-style accumulation for dist2 when available
                const mipp::Reg<T> r_dist2 =
                    mipp::fmadd(r_dx, r_dx,
                    mipp::fmadd(r_dy, r_dy,
                    mipp::fmadd(r_dz, r_dz, r_soft)));

                const mipp::Reg<T> r_inv  = mipp::rsqrt(r_dist2);
                const mipp::Reg<T> r_inv3 = r_inv * r_inv * r_inv;

                const mipp::Reg<T> r_fac = r_G * r_mj * r_inv3;

                r_ax = mipp::fmadd(r_fac, r_dx, r_ax);
                r_ay = mipp::fmadd(r_fac, r_dy, r_ay);
                r_az = mipp::fmadd(r_fac, r_dz, r_az);
            }
        }

        // Vector remainder
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

            const mipp::Reg<T> r_inv  = mipp::rsqrt(r_dist2);
            const mipp::Reg<T> r_inv3 = r_inv * r_inv * r_inv;

            const mipp::Reg<T> r_fac = r_G * r_mj * r_inv3;

            r_ax = mipp::fmadd(r_fac, r_dx, r_ax);
            r_ay = mipp::fmadd(r_fac, r_dy, r_ay);
            r_az = mipp::fmadd(r_fac, r_dz, r_az);
        }

        T ax = mipp::hadd(r_ax);
        T ay = mipp::hadd(r_ay);
        T az = mipp::hadd(r_az);

        // Scalar tail: keep math consistent with SIMD path (inv^3).
        for (long jt = n_vec; jt < n; jt++) {
            const T dx = qx[jt] - qi_x;
            const T dy = qy[jt] - qi_y;
            const T dz = qz[jt] - qi_z;

            const T dist2 = dx*dx + dy*dy + dz*dz + softSquared;

            const T inv  = inv_sqrt_scalar<T>(dist2);
            const T inv3 = inv * inv * inv;

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
