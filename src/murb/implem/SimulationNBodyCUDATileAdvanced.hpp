#ifndef SIMULATION_N_BODY_CUDA_TILE_HPP_
#define SIMULATION_N_BODY_CUDA_TILE_HPP_

#include <string>
#include <vector>

#include "core/SimulationNBodyInterface.hpp"
#include "core/Bodies.hpp"
#include "core/CUDABodies.hpp"

template <typename T>
class SimulationNBodyCUDATileAdvanced : public SimulationNBodyInterface<T> {
  protected:
    accSoA_t<T> accelerations; /*!< Array of body acceleration structures. */
    devAccSoA_t<T> devAccelerations;
    T* devM; 
    T* devQx;
    T* devQy;
    T* devQz;
    const T softSquared;
    int _num_threads, _num_blocks, _elem_per_thread;
  public:
    SimulationNBodyCUDATileAdvanced(const BodiesAllocatorInterface<T>& allocator, const T soft = 0.035f);
                         
    virtual ~SimulationNBodyCUDATileAdvanced();
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};


#endif /* SIMULATION_N_BODY_NAIVE_HPP_ */
