#include <mipp.h>

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyInterface.hpp"

SimulationNBodyInterface::SimulationNBodyInterface(const unsigned long nBodies, const std::string &scheme,
                                                   const float soft, const unsigned long randInit)
    : bodies(nBodies, scheme, randInit), dt(std::numeric_limits<float>::infinity()), soft(soft), flopsPerIte(0),
      allocatedBytes(bodies.getAllocatedBytes())
{
    this->allocatedBytes += (this->bodies.getN() + this->bodies.getPadding()) * sizeof(float) * 3;
}

const Bodies<float> &SimulationNBodyInterface::getBodies() const { return this->bodies; }

void SimulationNBodyInterface::setDt(float dtVal) { this->dt = dtVal; }

const float SimulationNBodyInterface::getDt() const { return this->dt; }

const float SimulationNBodyInterface::getFlopsPerIte() const { return this->flopsPerIte; }

const float SimulationNBodyInterface::getAllocatedBytes() const { return this->allocatedBytes; }
