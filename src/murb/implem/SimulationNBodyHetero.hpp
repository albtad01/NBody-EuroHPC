#ifndef SIMULATION_N_BODY_HETERO_HPP_
#define SIMULATION_N_BODY_HETERO_HPP_

#include <vector>
#include "core/SimulationNBodyInterface.hpp"

template <typename T>
class SimulationNBodyHetero : public SimulationNBodyInterface<T> {
protected:
    std::vector<accAoS_t<T>> accelerations;

public:
    SimulationNBodyHetero(const BodiesAllocatorInterface<T>& allocator, const T soft = (T)0.035);
    virtual ~SimulationNBodyHetero() = default;

    virtual void computeOneIteration();
    const std::vector<accAoS_t<T>>& getAccAoS();

protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif
