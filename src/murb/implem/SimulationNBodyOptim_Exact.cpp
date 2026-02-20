#include <algorithm>
#include <cmath>

#include "SimulationNBodyOptim_Exact.hpp"

template <typename T>
SimulationNBodyOptim_Exact<T>::SimulationNBodyOptim_Exact(const BodiesAllocatorInterface<T>& allocator, const T soft)
: SimulationNBodyInterface<T>(allocator, soft)
{
    this->flopsPerIte = 20.f * (T)this->getBodies()->getN() * (T)this->getBodies()->getN();
    accelerations.resize(this->getBodies()->getN());
}

template <typename T>
void SimulationNBodyOptim_Exact<T>::initIteration()
{
    // opzionale: se vuoi davvero "exact", azzera qui
    // std::fill(accelerations.begin(), accelerations.end(), accAoS_t<T>{T(0),T(0),T(0)});
}

template <typename T>
void SimulationNBodyOptim_Exact<T>::computeBodiesAcceleration()
{
    const auto& d = this->getBodies()->getDataSoA();
    const T* __restrict__ m  = d.m.data();
    const T* __restrict__ qx = d.qx.data();
    const T* __restrict__ qy = d.qy.data();
    const T* __restrict__ qz = d.qz.data();

    const long n = (long)this->getBodies()->getN();
    const T soft2 = this->soft * this->soft;
    const T G = this->G;

    auto* __restrict__ acc = accelerations.data();

    for (long i = 0; i < n; i++) {
        const T qi_x = qx[i];
        const T qi_y = qy[i];
        const T qi_z = qz[i];

        T ax = T(0), ay = T(0), az = T(0);

        for (long j = 0; j < n; j++) {
            const T dx = qx[j] - qi_x;
            const T dy = qy[j] - qi_y;
            const T dz = qz[j] - qi_z;

            const T dist2 = dx*dx + dy*dy + dz*dz + soft2;
            const T inv  = T(1) / std::sqrt(dist2);
            const T inv3 = inv * inv * inv;

            const T fac = G * m[j] * inv3;

            ax += fac * dx;
            ay += fac * dy;
            az += fac * dz;
        }

        acc[i].ax = ax;
        acc[i].ay = ay;
        acc[i].az = az;
    }
}

template <typename T>
void SimulationNBodyOptim_Exact<T>::computeOneIteration()
{
    initIteration();
    computeBodiesAcceleration();
    this->bodies->updatePositionsAndVelocities(accelerations, this->dt);
}

template class SimulationNBodyOptim_Exact<float>;
template class SimulationNBodyOptim_Exact<double>;
