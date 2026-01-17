#ifndef SIMULATION_N_BODY_CUDA_TILE_HPP_
#define SIMULATION_N_BODY_CUDA_TILE_HPP_

#include <string>
#include <vector>

#include "core/SimulationNBodyInterface.hpp"
#include "core/Bodies.hpp"

template <typename T>
class SimulationNBodyCUDATile : public SimulationNBodyInterface<T> {
  protected:
    std::vector<accAoS_t<T>> accelerations; /*!< Array of body acceleration structures. */
    accAoS_t<T>* devAccelerations;
    T* devM; 
    T* devQx;
    T* devQy;
    T* devQz;
    const T softSquared;
  public:
    SimulationNBodyCUDATile(const BodiesAllocatorInterface<T>& allocator, const T soft = 0.035f);
                         
    virtual ~SimulationNBodyCUDATile();
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};


#endif /* SIMULATION_N_BODY_NAIVE_HPP_ */
