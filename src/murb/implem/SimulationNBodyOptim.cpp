#include <cmath>
#include <vector>
#include <algorithm>
#include "SimulationNBodyOptim.hpp"

template <typename T>
SimulationNBodyOptim<T>::SimulationNBodyOptim(const BodiesAllocatorInterface<T>& allocator, const T soft)
    : SimulationNBodyInterface<T>(allocator, soft)
{
    const unsigned long n = this->getBodies()->getN();
    this->flopsPerIte = 20.f * (T)n * (T)n;

    this->accelerations.resize(n);
    this->temp_ax.resize(n);
    this->temp_ay.resize(n);
    this->temp_az.resize(n);
}

template <typename T>
void SimulationNBodyOptim<T>::initIteration()
{
    std::fill(this->temp_ax.begin(), this->temp_ax.end(), T(0));
    std::fill(this->temp_ay.begin(), this->temp_ay.end(), T(0));
    std::fill(this->temp_az.begin(), this->temp_az.end(), T(0));
}

template <typename T>
const std::vector<accAoS_t<T>>& SimulationNBodyOptim<T>::getAccAoS()
{
    return accelerations;
}

template <typename T>
void SimulationNBodyOptim<T>::computeBodiesAcceleration()
{
    const auto& d = this->getBodies()->getDataSoA();
    const T* __restrict__ qx = d.qx.data();
    const T* __restrict__ qy = d.qy.data();
    const T* __restrict__ qz = d.qz.data();
    const T* __restrict__ m  = d.m.data();

    const unsigned long n = this->getBodies()->getN();
    const T soft2 = this->soft * this->soft;
    const T G = this->G;

    T* __restrict__ ax = this->temp_ax.data();
    T* __restrict__ ay = this->temp_ay.data();
    T* __restrict__ az = this->temp_az.data();

    for (unsigned long i = 0; i < n; ++i) {
        const T xi = qx[i];
        const T yi = qy[i];
        const T zi = qz[i];
        const T mi = m[i];

        T axi = ax[i];
        T ayi = ay[i];
        T azi = az[i];

        for (unsigned long j = i + 1; j < n; ++j) {
            const T dx = qx[j] - xi;
            const T dy = qy[j] - yi;
            const T dz = qz[j] - zi;

            const T dist2 = dx*dx + dy*dy + dz*dz + soft2;

            const T inv  = T(1) / std::sqrt(dist2);
            const T inv3 = inv * inv * inv;
            const T s = G * inv3;

            const T mj = m[j];
            const T fij = s * mj;
            const T fji = s * mi;

            axi += fij * dx;
            ayi += fij * dy;
            azi += fij * dz;

            ax[j] -= fji * dx;
            ay[j] -= fji * dy;
            az[j] -= fji * dz;
        }

        ax[i] = axi;
        ay[i] = ayi;
        az[i] = azi;
    }

    for (unsigned long i = 0; i < n; ++i) {
        this->accelerations[i].ax = ax[i];
        this->accelerations[i].ay = ay[i];
        this->accelerations[i].az = az[i];
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
