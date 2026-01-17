#ifndef SIMULATION_N_BODY_CUDA_TILE_FULL_DEVICE_HPP_
#define SIMULATION_N_BODY_CUDA_TILE_FULL_DEVICE_HPP_

#include <string>
#include <vector>

#include "core/SimulationNBodyInterface.hpp"
#include "core/Bodies.hpp"

template <typename T>
class SimulationNBodyCUDATileFullDevice : public SimulationNBodyInterface<T> {
  protected:
    std::vector<accAoS_t<T>> accelerations; /*!< Array of body acceleration structures. */
    accAoS_t<T>* devAccelerations;
    T* devM; 
    T* devQx;
    T* devQy;
    T* devQz;
    const T softSquared;
  public:
    SimulationNBodyCUDATileFullDevice(const unsigned long nBodies, const std::string &scheme = "galaxy", const T soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyCUDATileFullDevice();
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();

    
};

#endif 
