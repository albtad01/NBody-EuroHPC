#ifndef SIMULATION_N_BODY_OPTIM_HPP_
#define SIMULATION_N_BODY_OPTIM_HPP_

#include <vector>
#include "core/SimulationNBodyInterface.hpp"

template <typename T>
class SimulationNBodyOptim : public SimulationNBodyInterface<T> {
  protected:
    std::vector<accAoS_t<T>> accelerations;

  public:
    SimulationNBodyOptim(const BodiesAllocatorInterface<T>& allocator, const T soft = T(0.035));
    virtual ~SimulationNBodyOptim() = default;

    virtual void computeOneIteration() override;
    const std::vector<accAoS_t<T>>& getAccAoS();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_OPTIM_HPP_ */