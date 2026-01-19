#ifndef SIMULATION_N_BODY_SIMD_HPP_
#define SIMULATION_N_BODY_SIMD_HPP_

#include <vector>
#include "core/SimulationNBodyInterface.hpp"

template <typename T>
class SimulationNBodySIMD : public SimulationNBodyInterface<T> {
  protected:
    std::vector<accAoS_t<T>> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodySIMD(const BodiesAllocatorInterface<T>& allocator, const T soft = 0.035f);
    virtual ~SimulationNBodySIMD() = default;
    virtual void computeOneIteration();
    const std::vector<accAoS_t<T>>& getAccAoS();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_SIMD_HPP_ */