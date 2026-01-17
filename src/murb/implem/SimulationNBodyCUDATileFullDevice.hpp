#ifndef SIMULATION_N_BODY_CUDA_TILE_FULL_DEVICE_HPP_
#define SIMULATION_N_BODY_CUDA_TILE_FULL_DEVICE_HPP_

#include <string>
#include <vector>

#include "core/SimulationNBodyInterface.hpp"
#include "core/Bodies.hpp"

class SimulationNBodyCUDATileFullDevice : public SimulationNBodyInterface<float> {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */
    accAoS_t<float>* devAccelerations;
    float* devM; 
    float* devQx;
    float* devQy;
    float* devQz;
    const float softSquared;
  public:
    SimulationNBodyCUDATileFullDevice(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyCUDATileFullDevice();
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();

    
};

#endif 
