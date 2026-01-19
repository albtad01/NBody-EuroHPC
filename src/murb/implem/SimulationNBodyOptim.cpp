#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyOptim.hpp"

// Fast inverse sqrt (use rsqrt for float, 1/sqrt for double)
template <typename T>
static inline T inv_sqrt(T x) {
    return T(1) / std::sqrt(x);
}
template <>
inline float inv_sqrt<float>(float x) {
    return rsqrtf(x);
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

    // Hoist constants out of loops
    const T softSquared = this->soft * this->soft;
    const T G = this->G;

    for (long i = 0; i < n; i++) {
        const T qi_x = qx[i];
        const T qi_y = qy[i];
        const T qi_z = qz[i];

        T ax = T(0), ay = T(0), az = T(0);

        // Reduce memory traffic: keep pointers local, simple loop
        for (long j = 0; j < n; j++) {
            const T dx = qx[j] - qi_x;
            const T dy = qy[j] - qi_y;
            const T dz = qz[j] - qi_z;

            const T dist2 = dx*dx + dy*dy + dz*dz + softSquared;

            // invDist^3 = (1/sqrt(dist2))^3 (float uses rsqrtf)
            const T inv  = inv_sqrt<T>(dist2);
            const T inv3 = inv * inv * inv;

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
