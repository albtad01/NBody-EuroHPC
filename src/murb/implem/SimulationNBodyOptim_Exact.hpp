#ifndef SIMULATION_N_BODY_OPTIM_EXACT_HPP_
#define SIMULATION_N_BODY_OPTIM_EXACT_HPP_

#include <vector>
#include "core/SimulationNBodyInterface.hpp"

template <typename T>
class SimulationNBodyOptim_Exact : public SimulationNBodyInterface<T> {
protected:
    std::vector<accAoS_t<T>> accelerations;

public:
    SimulationNBodyOptim_Exact(const BodiesAllocatorInterface<T>& allocator, const T soft = T(0.035));
    ~SimulationNBodyOptim_Exact() override = default;

    void computeOneIteration() override;
    const std::vector<accAoS_t<T>>& getAccAoS() { return accelerations; }

protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_OPTIM_EXACT_HPP_ */
