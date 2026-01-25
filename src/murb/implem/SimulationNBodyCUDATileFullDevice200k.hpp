#ifndef SIMULATION_N_BODY_CUDA_TILE_FULL_DEVICE_200K_HPP_
#define SIMULATION_N_BODY_CUDA_TILE_FULL_DEVICE_200K_HPP_

#include <string>
#include <vector>

#include "core/SimulationNBodyInterface.hpp"
#include "core/CUDABodies.hpp"

template <typename T>
class SimulationNBodyCUDATileFullDevice200k : public SimulationNBodyInterface<T> {
  protected:
    std::shared_ptr<CUDABodies<T>> cudaBodiesPtr;
    devAccSoA_t<T> devAccelerations;
    T* devGM;
    accSoA_t<T> accSoA;
    const T softSquared;
    bool const transfer_each_iteration;
    int _num_threads, _num_blocks;
  public:
    SimulationNBodyCUDATileFullDevice200k(const BodiesAllocatorInterface<T>& allocator, 
        const T soft = 0.035f, const bool transfer_each_iteration = false);
    virtual ~SimulationNBodyCUDATileFullDevice200k();
    virtual void computeOneIteration();
    const accSoA_t<T>& getAccSoA();

  protected:
    void initIteration();
    void computeBodiesAcceleration();

    
};

#endif 
