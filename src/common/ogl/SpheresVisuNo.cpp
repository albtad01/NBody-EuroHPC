#include "SpheresVisuNo.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

template <typename T> SpheresVisuNo<T>::SpheresVisuNo() : SpheresVisu() {}

template <typename T> SpheresVisuNo<T>::~SpheresVisuNo() {}

template <typename T> void SpheresVisuNo<T>::refreshDisplay() {}

template <typename T> bool SpheresVisuNo<T>::windowShouldClose() { return false; }

template <typename T> bool SpheresVisuNo<T>::pressedSpaceBar() { return false; }

template <typename T> bool SpheresVisuNo<T>::pressedPageUp() { return false; }

template <typename T> bool SpheresVisuNo<T>::pressedPageDown() { return false; }

// ==================================================================================== explicit template instantiation
template class SpheresVisuNo<double>;
template class SpheresVisuNo<float>;
// ==================================================================================== explicit template instantiation
