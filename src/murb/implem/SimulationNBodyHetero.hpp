#ifndef SIMULATION_N_BODY_HETERO_HPP_
#define SIMULATION_N_BODY_HETERO_HPP_

#include <vector>
#include <cstddef>
#include <cuda_runtime.h>
#include "core/SimulationNBodyInterface.hpp"

template <typename T>
class SimulationNBodyHetero : public SimulationNBodyInterface<T> {
protected:
    std::vector<accAoS_t<T>> accelerations;

    T* d_m  = nullptr;
    T* d_qx = nullptr;
    T* d_qy = nullptr;
    T* d_qz = nullptr;

    T* d_ax = nullptr;
    T* d_ay = nullptr;
    T* d_az = nullptr;

    cudaStream_t stream = nullptr;

    std::size_t cap_n   = 0;
    std::size_t cap_cut = 0;

    std::vector<T> hax;
    std::vector<T> hay;
    std::vector<T> haz;

public:
    SimulationNBodyHetero(const BodiesAllocatorInterface<T>& allocator, const T soft = (T)0.035);
    ~SimulationNBodyHetero() override;

    void computeOneIteration() override;
    const std::vector<accAoS_t<T>>& getAccAoS();

protected:
    void initIteration();
    void computeBodiesAcceleration();

    void ensureCuda(std::size_t n, std::size_t cut);
    void releaseCuda();
};

#endif
