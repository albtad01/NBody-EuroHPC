#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "mipp.h"
#include "SimulationNBodySIMD.hpp"

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
    const auto& bodiesData = this->getBodies()->getDataSoA();

    const T* __restrict__ m  = bodiesData.m.data();
    const T* __restrict__ qx = bodiesData.qx.data();
    const T* __restrict__ qy = bodiesData.qy.data();
    const T* __restrict__ qz = bodiesData.qz.data();

    const long n = (long)this->getBodies()->getN();
    const T softSquared = this->soft * this->soft;
    const T G = this->G;

    const int V = mipp::N<T>();
    const long n_vec = n - (n % V);

    for (long i = 0; i < n; i++) {

        const T qi_x = qx[i];
        const T qi_y = qy[i];
        const T qi_z = qz[i];

        mipp::Reg<T> r_ax = T(0);
        mipp::Reg<T> r_ay = T(0);
        mipp::Reg<T> r_az = T(0);

        const mipp::Reg<T> r_qi_x = qi_x;
        const mipp::Reg<T> r_qi_y = qi_y;
        const mipp::Reg<T> r_qi_z = qi_z;
        const mipp::Reg<T> r_soft = softSquared;
        const mipp::Reg<T> r_G    = G;

        for (long j = 0; j < n_vec; j += V) {

            const mipp::Reg<T> r_mj  = &m[j];
            const mipp::Reg<T> r_qjx = &qx[j];
            const mipp::Reg<T> r_qjy = &qy[j];
            const mipp::Reg<T> r_qjz = &qz[j];

            const mipp::Reg<T> r_dx = r_qjx - r_qi_x;
            const mipp::Reg<T> r_dy = r_qjy - r_qi_y;
            const mipp::Reg<T> r_dz = r_qjz - r_qi_z;

            const mipp::Reg<T> r_dist2 = r_dx*r_dx + r_dy*r_dy + r_dz*r_dz + r_soft;

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

        // Scalar tail: avoid pow(), keep same math as SIMD path.
        for (long j = n_vec; j < n; j++) {
            const T dx = qx[j] - qi_x;
            const T dy = qy[j] - qi_y;
            const T dz = qz[j] - qi_z;

            const T dist2 = dx*dx + dy*dy + dz*dz + softSquared;

            const T inv  = T(1) / std::sqrt(dist2);
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
void SimulationNBodySIMD<T>::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    this->bodies->updatePositionsAndVelocities(this->accelerations, this->dt);
}

template class SimulationNBodySIMD<float>;
template class SimulationNBodySIMD<double>;