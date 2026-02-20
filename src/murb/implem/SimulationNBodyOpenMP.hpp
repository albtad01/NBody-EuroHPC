#ifndef SIMULATION_N_BODY_OPENMP_HPP_
#define SIMULATION_N_BODY_OPENMP_HPP_

#include <vector>
#include "core/SimulationNBodyInterface.hpp"

template <typename T>
class SimulationNBodyOpenMP : public SimulationNBodyInterface<T> { 
  protected:
    std::vector<accAoS_t<T>> accelerations; 

  public:
    // Costruttore corretto
    SimulationNBodyOpenMP(const BodiesAllocatorInterface<T>& allocator, const T soft = 0.035f);
    virtual ~SimulationNBodyOpenMP() = default;
    virtual void computeOneIteration();
    const std::vector<accAoS_t<T>>& getAccAoS();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif