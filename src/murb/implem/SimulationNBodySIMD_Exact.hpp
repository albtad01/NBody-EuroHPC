#ifndef SIMULATION_N_BODY_SIMD_EXACT_HPP_
#define SIMULATION_N_BODY_SIMD_EXACT_HPP_

#include <vector>
#include "core/SimulationNBodyInterface.hpp"

template <typename T>
class SimulationNBodySIMD_Exact : public SimulationNBodyInterface<T> {
protected:
    std::vector<accAoS_t<T>> accelerations;

public:
    SimulationNBodySIMD_Exact(const BodiesAllocatorInterface<T>& allocator, const T soft = T(0.035));
    ~SimulationNBodySIMD_Exact() override = default;

    void computeOneIteration() override;
    const std::vector<accAoS_t<T>>& getAccAoS();

protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_SIMD_EXACT_HPP_ */
