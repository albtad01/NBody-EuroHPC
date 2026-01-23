#ifndef HISTORY_TRACKING_INTERFACE_CPP_
#define HISTORY_TRACKING_INTERFACE_CPP_
#include "HistoryTrackingInterface.hpp"

template <typename T>
HistoryTrackingInterface<T>::HistoryTrackingInterface(SimulationHistory<T>& history) 
    : history{history}
{}

template <typename T>
const SimulationHistory<T>& HistoryTrackingInterface<T>::getHistory() const {
    return this->history;
}

template <typename T>
GPUHistoryTrackingInterface<T>::GPUHistoryTrackingInterface(GPUSimulationHistory<T>& history) 
    : history{history}
{}

template <typename T>
const GPUSimulationHistory<T>& GPUHistoryTrackingInterface<T>::getHistory() const {
    return this->history;
}

template class HistoryTrackingInterface<float>;
template class GPUHistoryTrackingInterface<float>;

#endif