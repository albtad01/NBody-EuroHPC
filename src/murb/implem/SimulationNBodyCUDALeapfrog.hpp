#ifndef SIMULATION_N_BODY_CUDA_LEAPFROG_HPP_
#define SIMULATION_N_BODY_CUDA_LEAPFROG_HPP_

#include <string>
#include <vector>

#include "core/SimulationNBodyInterface.hpp"
#include "core/CUDABodies.hpp"
#include "core/HistoryTrackingInterface.hpp"

template <typename T, typename Q=T>
class SimulationNBodyCUDALeapfrog : public SimulationNBodyInterface<T>, public GPUHistoryTrackingInterface<Q> {
  protected:
    std::shared_ptr<CUDABodies<T>> cudaBodiesPtr;
    devAccSoA_t<T> devAccelerations;
    T* devGM;
    accSoA_t<T> accSoA;
    const T softSquared;
    bool const transfer_each_iteration;
    int _num_threads, _num_blocks, _elem_per_thread;

    Q* bufferForEnergy;

    int currentIteration = -1;
    const int numIterations;
  public:
    SimulationNBodyCUDALeapfrog(const BodiesAllocatorInterface<T>& allocator, 
        std::shared_ptr<GPUSimulationHistory<Q>> history, 
        int NIterations,
        const T soft = 0.035f, const bool transfer_each_iteration = false);
    virtual ~SimulationNBodyCUDALeapfrog();
    virtual void computeOneIteration();
    virtual void computeMetrics();
    const accSoA_t<T>& getAccSoA();

  protected:
    void initIteration();
    void computeBodiesAcceleration();

    
};

#endif 
