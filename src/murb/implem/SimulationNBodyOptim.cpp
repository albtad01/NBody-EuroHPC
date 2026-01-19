#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyOptim.hpp"

template <typename T>
static inline T inv_sqrt_scalar(T x) {
    // CPU portable: compilers usually map 1/sqrt(x) to a fast reciprocal-sqrt sequence.
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
    // Not needed: we overwrite accelerations[i] each iteration.
    // Kept empty on purpose to avoid extra memory traffic.
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

    const T softSquared = this->soft * this->soft; // hoist invariant
    const T G = this->G;                           // hoist invariant

    for (long i = 0; i < n; i++) {

        const T qi_x = qx[i];
        const T qi_y = qy[i];
        const T qi_z = qz[i];

        T ax = T(0), ay = T(0), az = T(0);

        // Unroll by 4: usually good ILP without crazy register pressure.
        long j = 0;
        for (; j + 3 < n; j += 4) {
            // j+0
            const T dx0 = qx[j+0] - qi_x;
            const T dy0 = qy[j+0] - qi_y;
            const T dz0 = qz[j+0] - qi_z;
            const T d20 = dx0*dx0 + dy0*dy0 + dz0*dz0 + softSquared;
            const T inv0 = inv_sqrt_scalar<T>(d20);
            const T inv30 = inv0 * inv0 * inv0;
            const T fac0 = G * m[j+0] * inv30;
            ax += fac0 * dx0; ay += fac0 * dy0; az += fac0 * dz0;

            // j+1
            const T dx1 = qx[j+1] - qi_x;
            const T dy1 = qy[j+1] - qi_y;
            const T dz1 = qz[j+1] - qi_z;
            const T d21 = dx1*dx1 + dy1*dy1 + dz1*dz1 + softSquared;
            const T inv1 = inv_sqrt_scalar<T>(d21);
            const T inv31 = inv1 * inv1 * inv1;
            const T fac1 = G * m[j+1] * inv31;
            ax += fac1 * dx1; ay += fac1 * dy1; az += fac1 * dz1;

            // j+2
            const T dx2 = qx[j+2] - qi_x;
            const T dy2 = qy[j+2] - qi_y;
            const T dz2 = qz[j+2] - qi_z;
            const T d22 = dx2*dx2 + dy2*dy2 + dz2*dz2 + softSquared;
            const T inv2 = inv_sqrt_scalar<T>(d22);
            const T inv32 = inv2 * inv2 * inv2;
            const T fac2 = G * m[j+2] * inv32;
            ax += fac2 * dx2; ay += fac2 * dy2; az += fac2 * dz2;

            // j+3
            const T dx3 = qx[j+3] - qi_x;
            const T dy3 = qy[j+3] - qi_y;
            const T dz3 = qz[j+3] - qi_z;
            const T d23 = dx3*dx3 + dy3*dy3 + dz3*dz3 + softSquared;
            const T inv3 = inv_sqrt_scalar<T>(d23);
            const T inv33 = inv3 * inv3 * inv3;
            const T fac3 = G * m[j+3] * inv33;
            ax += fac3 * dx3; ay += fac3 * dy3; az += fac3 * dz3;
        }

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
