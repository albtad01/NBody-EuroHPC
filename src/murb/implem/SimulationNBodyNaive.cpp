#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyNaive.hpp"

template <typename T>
SimulationNBodyNaive<T>::SimulationNBodyNaive(const unsigned long nBodies, const std::string &scheme, const T soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface<T>(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (T)this->getBodies().getN() * (T)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());
}

template <typename T>
void SimulationNBodyNaive<T>::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

template <typename T>
void SimulationNBodyNaive<T>::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<T>> &d = this->getBodies().getDataAoS();

    // flops = n² * 20
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        // flops = n * 20
        for (unsigned long jBody = 0; jBody < this->getBodies().getN(); jBody++) {
            const T rijx = d[jBody].qx - d[iBody].qx; // 1 flop
            const T rijy = d[jBody].qy - d[iBody].qy; // 1 flop
            const T rijz = d[jBody].qz - d[iBody].qz; // 1 flop

            // compute the || rij ||² distance between body i and body j
            const T rijSquared = std::pow(rijx, 2) + std::pow(rijy, 2) + std::pow(rijz, 2); // 5 flops
            // compute e²
            const T softSquared = std::pow(this->soft, 2); // 1 flops
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            const T ai = this->G * d[jBody].m / std::pow(rijSquared + softSquared, 3.f / 2.f); // 5 flops

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            this->accelerations[iBody].ax += ai * rijx; // 2 flops
            this->accelerations[iBody].ay += ai * rijy; // 2 flops
            this->accelerations[iBody].az += ai * rijz; // 2 flops
        }
    }
}

template <typename T>
void SimulationNBodyNaive<T>::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}


SimulationNBodyNaive<float> srf(10, "random", 1e-4);
SimulationNBodyNaive<double> srd(10, "random", 1e-4);
SimulationNBodyNaive<float> sgf(10, "galaxy", 1e-4);
SimulationNBodyNaive<double> sgd(10, "galaxy", 1e-4);