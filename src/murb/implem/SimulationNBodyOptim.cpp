#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyOptim.hpp"

template <typename T>
static inline T inv_sqrt_scalar(T x) {
    // Portable scalar fallback. (No rsqrtf on CPU.)
    return T(1) / std::sqrt(x);
}

template <typename T>
SimulationNBodyOptim<T>::SimulationNBodyOptim(const BodiesAllocatorInterface<T>& allocator, const T soft)
    : SimulationNBodyInterface<T>(allocator, soft)
{
    this->flopsPerIte = 20.f * (T)this->getBodies()->getN() * (T)this->getBodies()->getN();
    this->accelerations.resize(this->getBodies()->getN());
}

template <typename T>
void SimulationNBodyOptim<T>::initIteration()
{
    std::fill(this->accelerations.begin(), this->accelerations.end(), accAoS_t<T>{T(0), T(0), T(0)});
}

template <typename T>
const std::vector<accAoS_t<T>>& SimulationNBodyOptim<T>::getAccAoS() {
    return accelerations;
}

template <typename T>
void SimulationNBodyOptim<T>::computeBodiesAcceleration()
{
    const auto& d = this->getBodies()->getDataSoA();
    const T* __restrict__ m  = d.m.data();
    const T* __restrict__ qx = d.qx.data();
    const T* __restrict__ qy = d.qy.data();
    const T* __restrict__ qz = d.qz.data();

    const long n = (long)this->getBodies()->getN();

    const T softSquared = this->soft * this->soft;  // hoist invariant
    const T G = this->G;                            // hoist invariant

    for (long i = 0; i < n; i++) {
        const T qi_x = qx[i];
        const T qi_y = qy[i];
        const T qi_z = qz[i];

        T ax = T(0), ay = T(0), az = T(0);

        // Small unroll to increase ILP without too much register pressure.
        long j = 0;
        for (; j + 1 < n; j += 2) {
            // j
            const T dx0 = qx[j]   - qi_x;
            const T dy0 = qy[j]   - qi_y;
            const T dz0 = qz[j]   - qi_z;
            const T dist20 = dx0*dx0 + dy0*dy0 + dz0*dz0 + softSquared;
            const T inv0   = inv_sqrt_scalar<T>(dist20);
            const T inv30  = inv0 * inv0 * inv0;
            const T fac0   = G * m[j] * inv30;
            ax += fac0 * dx0;
            ay += fac0 * dy0;
            az += fac0 * dz0;

            // j+1
            const T dx1 = qx[j+1] - qi_x;
            const T dy1 = qy[j+1] - qi_y;
            const T dz1 = qz[j+1] - qi_z;
            const T dist21 = dx1*dx1 + dy1*dy1 + dz1*dz1 + softSquared;
            const T inv1   = inv_sqrt_scalar<T>(dist21);
            const T inv31  = inv1 * inv1 * inv1;
            const T fac1   = G * m[j+1] * inv31;
            ax += fac1 * dx1;
            ay += fac1 * dy1;
            az += fac1 * dz1;
        }

        // tail (if n is odd)
        for (; j < n; j++) {
            const T dx = qx[j] - qi_x;
            const T dy = qy[j] - qi_y;
            const T dz = qz[j] - qi_z;

            const T dist2 = dx*dx + dy*dy + dz*dz + softSquared;
            const T inv   = inv_sqrt_scalar<T>(dist2);
            const T inv3  = inv * inv * inv;

            const T fac = G * m[j] * inv3;

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
void SimulationNBodyOptim<T>::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    this->bodies->updatePositionsAndVelocities(this->accelerations, this->dt);
}

template class SimulationNBodyOptim<float>;
template class SimulationNBodyOptim<double>;
