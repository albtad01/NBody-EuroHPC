#ifndef SIMULATION_N_BODY_INTERFACE_CPP_
#define SIMULATION_N_BODY_INTERFACE_CPP_

#include <cassert>
#include <fstream>
#include <limits>

#include "SimulationNBodyInterface.hpp"

template <typename T>
SimulationNBodyInterface<T>::SimulationNBodyInterface(const unsigned long nBodies, const std::string &scheme,
                                                   const T soft, const unsigned long randInit)
    : bodies(nBodies, scheme, randInit), dt(std::numeric_limits<T>::infinity()), soft(soft), flopsPerIte(0),
      allocatedBytes(bodies.getAllocatedBytes())
{
    this->allocatedBytes += (this->bodies.getN() + this->bodies.getPadding()) * sizeof(T) * 3;
}

template <typename T>
const Bodies<T> &SimulationNBodyInterface<T>::getBodies() const { return this->bodies; }

template <typename T>
void SimulationNBodyInterface<T>::setDt(T dtVal) { this->dt = dtVal; }

template <typename T>
const T SimulationNBodyInterface<T>::getDt() const { return this->dt; }

template <typename T>
const T SimulationNBodyInterface<T>::getFlopsPerIte() const { return this->flopsPerIte; }

template <typename T>
const T SimulationNBodyInterface<T>::getAllocatedBytes() const { return this->allocatedBytes; }


template class SimulationNBodyInterface<float>;
template class SimulationNBodyInterface<double>;

#endif