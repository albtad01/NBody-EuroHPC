#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>

#include "SimulationNBodyNaiveCadna.hpp"

//#include <cadna.h>

template <typename T>                          
SimulationNBodyNaiveCadna<T>::SimulationNBodyNaiveCadna(const BodiesAllocatorInterface<T>& allocator, const T soft)
    : SimulationNBodyInterface<T>(allocator, soft)
{
    this->flopsPerIte = 20.f * (T)this->getBodies()->getN() * (T)this->getBodies()->getN();
    this->accelerations.resize(this->getBodies()->getN());

    //cadna_init(-1);
}

template <typename T>
void SimulationNBodyNaiveCadna<T>::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies()->getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

template <typename T>
const std::vector<accAoS_t<T>>& SimulationNBodyNaiveCadna<T>::getAccAoS() {
    return accelerations;
}
template <typename T>
void SimulationNBodyNaiveCadna<T>::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<T>> &d = this->getBodies()->getDataAoS();

    // flops = n² * 20
    for (unsigned long iBody = 0; iBody < this->getBodies()->getN(); iBody++) {
        // flops = n * 20
        for (unsigned long jBody = 0; jBody < this->getBodies()->getN(); jBody++) {
            const T rijx = d[jBody].qx - d[iBody].qx; // 1 flop
            const T rijy = d[jBody].qy - d[iBody].qy; // 1 flop
            const T rijz = d[jBody].qz - d[iBody].qz; // 1 flop

            // compute the || rij ||² distance between body i and body j
            const T inv = 1 / sqrt(rijx * rijx + rijy * rijy + rijz * rijz); // 5 flops
            // compute e²
            const T softSquared = this->soft * this->soft; // 1 flops
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            const T ai = this->G * d[jBody].m * inv * inv * inv; // 5 flops

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            this->accelerations[iBody].ax += ai * rijx; // 2 flops
            this->accelerations[iBody].ay += ai * rijy; // 2 flops
            this->accelerations[iBody].az += ai * rijz; // 2 flops
        }
    }
}

template <typename T>
void SimulationNBodyNaiveCadna<T>::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies->updatePositionsAndVelocities(this->accelerations, this->dt);
    const auto &d = this->bodies->getDataAoS();
    // for ( const dataAoS_t<T> &data : d ) {
    //     printf("x: %s y: %s z: %s || ", strp(data.qx), strp(data.qy), strp(data.qz));
    // }
    printf("\n\n");
}


template <typename T>
SimulationNBodyNaiveCadna<T>::~SimulationNBodyNaiveCadna() {
    //cadna_end();
}

// template class SimulationNBodyNaiveCadna<float_st>;
// template class SimulationNBodyNaiveCadna<double_st>;
template class SimulationNBodyNaiveCadna<float>;
template class SimulationNBodyNaiveCadna<double>;