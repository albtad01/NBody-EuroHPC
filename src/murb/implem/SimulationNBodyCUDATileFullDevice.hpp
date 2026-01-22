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
    bool const transfer_each_iteration;
    int _num_threads, _num_blocks, _elem_per_thread;
  public:
    SimulationNBodyCUDATileFullDevice(const BodiesAllocatorInterface<T>& allocator, 
        const T soft = 0.035f, const bool transfer_each_iteration = false);
    virtual ~SimulationNBodyCUDATileFullDevice();
    virtual void computeOneIteration();
    const accSoA_t<T>& getAccSoA();

  protected:
    void initIteration();
    void computeBodiesAcceleration();

    
};

#endif 
