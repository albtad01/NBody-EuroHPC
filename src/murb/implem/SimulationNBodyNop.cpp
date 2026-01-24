#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>

#include "SimulationNBodyNop.hpp"

template <typename T>                          
SimulationNBodyNop<T>::SimulationNBodyNop(const BodiesAllocatorInterface<T>& allocator, const T soft)
    : SimulationNBodyInterface<T>(allocator, soft)
{
    this->flopsPerIte = 20.f * (T)this->getBodies()->getN() * (T)this->getBodies()->getN();
    this->accelerations.resize(this->getBodies()->getN());
}

template <typename T>
void SimulationNBodyNop<T>::initIteration()
{
}

template <typename T>
const std::vector<accAoS_t<T>>& SimulationNBodyNop<T>::getAccAoS() {
    return accelerations;
}
template <typename T>
void SimulationNBodyNop<T>::computeBodiesAcceleration()
{
}

template <typename T>
void SimulationNBodyNop<T>::computeOneIteration()
{
}


template class SimulationNBodyNop<float>;
template class SimulationNBodyNop<double>;