#ifndef SIMULATION_N_BODY_CUDA_TILE_HPP_
#define SIMULATION_N_BODY_CUDA_TILE_HPP_

#include <string>
#include <vector>

#include "core/SimulationNBodyInterface.hpp"
#include "core/Bodies.hpp"

class SimulationNBodyCUDATile : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */
    accAoS_t<float>* devAccelerations;
    float* devM; 
    float* devQx;
    float* devQy;
    float* devQz;
    const float softSquared;
  public:
    SimulationNBodyCUDATile(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyCUDATile();
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_NAIVE_HPP_ */
