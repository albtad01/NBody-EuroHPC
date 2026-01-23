#ifndef SIMULATION_N_BODY_CUDA_PROPERTY_TRACKING_HPP_
#define SIMULATION_N_BODY_CUDA_PROPERTY_TRACKING_HPP_

#include <string>
#include <vector>

#include "core/SimulationNBodyInterface.hpp"
#include "core/CUDABodies.hpp"
#include "core/HistoryTrackingInterface.hpp"

template <typename T>
class SimulationNBodyCUDAPropertyTracking : public SimulationNBodyInterface<T>, GPUHistoryTrackingInterface<T> {
  protected:
    std::shared_ptr<CUDABodies<T>> cudaBodiesPtr;
    devAccSoA_t<T> devAccelerations;
    T* devGM;
    accSoA_t<T> accSoA;
    const T softSquared;
    bool const transfer_each_iteration;
    int _num_threads, _num_blocks, _elem_per_thread;
  public:
    SimulationNBodyCUDAPropertyTracking(const BodiesAllocatorInterface<T>& allocator, 
        GPUSimulationHistory<T>& history, 
        const T soft = 0.035f, const bool transfer_each_iteration = false);
    virtual ~SimulationNBodyCUDAPropertyTracking();
    virtual void computeOneIteration();
    virtual void computeMetrics();
    const accSoA_t<T>& getAccSoA();

  protected:
    void initIteration();
    void computeBodiesAcceleration();

    
};

#endif 
