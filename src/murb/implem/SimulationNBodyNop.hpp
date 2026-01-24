#ifndef SIMULATION_N_BODY_NOP_HPP_
#define SIMULATION_N_BODY_NOP_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"

template <typename T>
class SimulationNBodyNop : public SimulationNBodyInterface<T> {
  protected:
    std::vector<accAoS_t<T>> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodyNop(const BodiesAllocatorInterface<T>& allocator, const T soft = 0.035f);
    virtual ~SimulationNBodyNop() = default;
    virtual void computeOneIteration();
    const std::vector<accAoS_t<T>>& getAccAoS();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_NAIVE_HPP_ */
