#ifndef HISTORY_TRACKING_INTERFACE_CPP_
#define HISTORY_TRACKING_INTERFACE_CPP_
#include "HistoryTrackingInterface.hpp"

template <typename T>
HistoryTrackingInterface<T>::HistoryTrackingInterface(std::shared_ptr<SimulationHistory<T>> _history) 
    : history{_history}
{}

template <typename T>
const std::shared_ptr<SimulationHistory<T>> HistoryTrackingInterface<T>::getHistory() const {
    return this->history;
}

template <typename T>
GPUHistoryTrackingInterface<T>::GPUHistoryTrackingInterface(std::shared_ptr<GPUSimulationHistory<T>> _history) 
    : history{_history}
{}

template <typename T>
const std::shared_ptr<GPUSimulationHistory<T>> GPUHistoryTrackingInterface<T>::getHistory() const {
    return this->history;
}

template class HistoryTrackingInterface<float>;
template class GPUHistoryTrackingInterface<float>;
template class HistoryTrackingInterface<double>;
template class GPUHistoryTrackingInterface<double>;

#endif