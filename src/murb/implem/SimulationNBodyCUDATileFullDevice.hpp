#ifndef SIMULATION_N_BODY_CUDA_TILE_FULL_DEVICE_HPP_
#define SIMULATION_N_BODY_CUDA_TILE_FULL_DEVICE_HPP_

#include <string>
#include <vector>

#include "core/SimulationNBodyInterface.hpp"
#include "core/CUDABodies.hpp"

template <typename T>
class SimulationNBodyCUDATileFullDevice : public SimulationNBodyInterface<T> {
  protected:
    std::shared_ptr<CUDABodies<T>> cudaBodiesPtr;
    devAccSoA_t<T> devAccelerations;
    accSoA_t<T> accSoA;
    const T softSquared;
  public:
    SimulationNBodyCUDATileFullDevice(const BodiesAllocatorInterface<T>& allocator, const T soft = 0.035f);
    virtual ~SimulationNBodyCUDATileFullDevice();
    virtual void computeOneIteration();
    const accSoA_t<T>& getAccSoA();

  protected:
    void initIteration();
    void computeBodiesAcceleration();

    
};

#endif 
