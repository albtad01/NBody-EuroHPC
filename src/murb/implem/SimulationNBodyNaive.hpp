#ifndef SIMULATION_N_BODY_NAIVE_HPP_
#define SIMULATION_N_BODY_NAIVE_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"

template <typename T>
class SimulationNBodyNaive : public SimulationNBodyInterface<T> {
  protected:
    std::vector<accAoS_t<T>> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodyNaive(const BodiesAllocatorInterface<T>& allocator, const T soft = 0.035f);
    virtual ~SimulationNBodyNaive() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_NAIVE_HPP_ */
