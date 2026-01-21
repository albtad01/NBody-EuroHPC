#ifndef SIMULATION_N_BODY_OPENMP_GREEN_HPP_
#define SIMULATION_N_BODY_OPENMP_GREEN_HPP_

#include <vector>
#include "core/SimulationNBodyInterface.hpp"


template <typename T>
class SimulationNBodyOpenMP_Green : public SimulationNBodyInterface<T> {
protected:
    std::vector<accAoS_t<T>> accelerations;

public:
    SimulationNBodyOpenMP_Green(const BodiesAllocatorInterface<T>& allocator, const T soft = T(0.035));
    ~SimulationNBodyOpenMP_Green() override = default;

    void computeOneIteration() override;
    const std::vector<accAoS_t<T>>& getAccAoS();

protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_OPENMP_GREEN_HPP_ */
