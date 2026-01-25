#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>

#include "SimulationNBodyNaive.hpp"

template <typename T>                          
SimulationNBodyNaive<T>::SimulationNBodyNaive(const BodiesAllocatorInterface<T>& allocator, const T soft)
    : SimulationNBodyInterface<T>(allocator, soft)
{
    this->flopsPerIte = 20.f * (T)this->getBodies()->getN() * (T)this->getBodies()->getN();
    this->accelerations.resize(this->getBodies()->getN());
}

template <typename T>
void SimulationNBodyNaive<T>::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies()->getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

template <typename T>
const std::vector<accAoS_t<T>>& SimulationNBodyNaive<T>::getAccAoS() {
    return accelerations;
}
template <typename T>
void SimulationNBodyNaive<T>::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<T>> &d = this->getBodies()->getDataAoS();

    for (unsigned long iBody = 0; iBody < this->getBodies()->getN(); iBody++) {
        for (unsigned long jBody = 0; jBody < this->getBodies()->getN(); jBody++) {
            const T rijx = d[jBody].qx - d[iBody].qx; 
            const T rijy = d[jBody].qy - d[iBody].qy;
            const T rijz = d[jBody].qz - d[iBody].qz; 

            const T rijSquared = std::pow(rijx, 2) + std::pow(rijy, 2) + std::pow(rijz, 2); // 5 flops
            const T softSquared = std::pow(this->soft, 2); 
            const T ai = this->G * d[jBody].m / std::pow(rijSquared + softSquared, 3.f / 2.f); 

            this->accelerations[iBody].ax += ai * rijx;
            this->accelerations[iBody].ay += ai * rijy;
            this->accelerations[iBody].az += ai * rijz; 
        }
    }
}

template <typename T>
void SimulationNBodyNaive<T>::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    this->bodies->updatePositionsAndVelocities(this->accelerations, this->dt);
}


template class SimulationNBodyNaive<float>;
template class SimulationNBodyNaive<double>;