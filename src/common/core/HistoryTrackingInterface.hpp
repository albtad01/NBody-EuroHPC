#ifndef HISTORY_TRACKING_INTERFACE_HPP_
#define HISTORY_TRACKING_INTERFACE_HPP_
#include "SimulationHistory.hpp"
#include "SimulationHistoryGPU.hpp"

template <typename T>
class HistoryTrackingInterface {
    protected:
        SimulationHistory<T>& history;
    public:
        HistoryTrackingInterface(SimulationHistory<T>& history);
        const SimulationHistory<T>& getHistory() const;
};

template <typename T>
class GPUHistoryTrackingInterface {
    protected:
        GPUSimulationHistory<T>& history;
    public:
        GPUHistoryTrackingInterface(GPUSimulationHistory<T>& history);
        const GPUSimulationHistory<T>& getHistory() const;
};

#endif