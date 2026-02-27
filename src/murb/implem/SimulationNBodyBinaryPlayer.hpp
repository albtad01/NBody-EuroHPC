#ifndef SIMULATION_N_BODY_BINARY_PLAYER_HPP_
#define SIMULATION_N_BODY_BINARY_PLAYER_HPP_

#include "core/SimulationNBodyInterface.hpp"
#include <fstream>
#include <string>

template <typename T>
class SimulationNBodyBinaryPlayer : public SimulationNBodyInterface<T> {
protected:
    std::ifstream inFile;
    std::string fileName;

public:
    SimulationNBodyBinaryPlayer(const BodiesAllocatorInterface<T>& allocator, const T soft, std::string filename = "simulation_data.bin");
    virtual ~SimulationNBodyBinaryPlayer();

    virtual void computeOneIteration() override;
};

#endif