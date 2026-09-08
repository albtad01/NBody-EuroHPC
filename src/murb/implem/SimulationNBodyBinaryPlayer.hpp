#ifndef SIMULATION_N_BODY_BINARY_PLAYER_HPP_
#define SIMULATION_N_BODY_BINARY_PLAYER_HPP_

#include "core/SimulationNBodyInterface.hpp"
#include "core/TrajectoryBinary.hpp"
#include <string>

template <typename T>
class SimulationNBodyBinaryPlayer : public SimulationNBodyInterface<T> {
protected:
    std::string fileName;
    murb::TrajectoryReader reader;

public:
    SimulationNBodyBinaryPlayer(const BodiesAllocatorInterface<T>& allocator, const T soft,
                                const std::string& filename);
    virtual ~SimulationNBodyBinaryPlayer() = default;

    virtual void computeOneIteration() override;
    bool readNextFrame();
    const murb::TrajectoryMetadata& getMetadata() const;
};

#endif
