#ifndef HISTORY_TRACKING_INTERFACE_HPP_
#define HISTORY_TRACKING_INTERFACE_HPP_

#include <memory>

#include "SimulationHistory.hpp"
#ifdef USE_CUDA
#include "SimulationHistoryGPU.hpp"
#endif

template <typename T>
class HistoryTrackingInterface {
    protected:
        std::shared_ptr<SimulationHistory<T>> history;
    public:
        HistoryTrackingInterface(std::shared_ptr<SimulationHistory<T>> history);
        const std::shared_ptr<SimulationHistory<T>> getHistory() const;
};

#ifdef USE_CUDA
template <typename T>
class GPUHistoryTrackingInterface {
    protected:
        std::shared_ptr<GPUSimulationHistory<T>> history;
    public:
        GPUHistoryTrackingInterface(std::shared_ptr<GPUSimulationHistory<T>> history);
        const std::shared_ptr<GPUSimulationHistory<T>> getHistory() const;
};
#endif

#endif