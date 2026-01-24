#include "SimulationHistoryGPU.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

#define CUDA_CHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

template <typename T>
inline std::vector<T> sum_in_place(std::vector<T>& dst, const std::vector<T>& src) {
    for (int i=0; i<src.size(); i++) {
        dst[i] += src[i];
    }
    return dst;
}

template <typename T>
inline std::vector<std::array<T,3>> sum_in_place(std::vector<std::array<T,3>>& dst, const std::vector<std::array<T,3>>& src) {
    for (int i=0; i<src.size(); i++) {
        dst[i][0] += src[i][0];
        dst[i][1] += src[i][1];
        dst[i][2] += src[i][2];
    }
    return dst;
}

// ========================= GPUSimulationHistoryInterface ============================
template <typename T>
void GPUSimulationHistoryInterface<T>::allocateDeviceMemory(int numIterations) {
    if (allocatedIterations == numIterations && devEnergies != nullptr) {
        return;  // Already allocated with correct size
    }

    deallocateDeviceMemory();  // Deallocate if present

    allocatedIterations = numIterations;
    CUDA_CHECK(cudaMalloc(&devEnergies, numIterations * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&devAngMomentums, numIterations * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&devDensityCentersX, numIterations * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&devDensityCentersY, numIterations * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&devDensityCentersZ, numIterations * sizeof(T)));
}

template <typename T>
void GPUSimulationHistoryInterface<T>::deallocateDeviceMemory() {
    if (devEnergies != nullptr) {
        cudaFree(devEnergies);
        devEnergies = nullptr;
    }
    if (devAngMomentums != nullptr) {
        cudaFree(devAngMomentums);
        devAngMomentums = nullptr;
    }
    if (devDensityCentersX != nullptr) {
        cudaFree(devDensityCentersX);
        devDensityCentersX = nullptr;
    }
    if (devDensityCentersY != nullptr) {
        cudaFree(devDensityCentersY);
        devDensityCentersY = nullptr;
    }
    if (devDensityCentersZ != nullptr) {
        cudaFree(devDensityCentersZ);
        devDensityCentersZ = nullptr;
    }
    allocatedIterations = 0;
}

template <typename T>
T* GPUSimulationHistoryInterface<T>::getDevEnergy() {
    return devEnergies;
}

template <typename T>
T* GPUSimulationHistoryInterface<T>::getDevAngMomentum() {
    return devAngMomentums;
}

template <typename T>
T* GPUSimulationHistoryInterface<T>::getDevDensityCentersX() {
    return devDensityCentersX;
}

template <typename T>
T* GPUSimulationHistoryInterface<T>::getDevDensityCentersY() {
    return devDensityCentersY;
}

template <typename T>
T* GPUSimulationHistoryInterface<T>::getDevDensityCentersZ() {
    return devDensityCentersZ;
}

template <typename T>
GPUSimulationHistoryInterface<T>::~GPUSimulationHistoryInterface() {
    deallocateDeviceMemory();
}

// =============================== GPUSimulationHistory =================================
// template <typename T>
// T* GPUSimulationHistory<T>::getDevEnergy() {
//     return GPUSimulationHistoryInterface<T>::getDevEnergy();
// }

// template <typename T>
// T* GPUSimulationHistory<T>::getDevAngMomentum() {
//     return GPUSimulationHistoryInterface<T>::getDevAngMomentum();
// }

// template <typename T>
// T* GPUSimulationHistory<T>::getDevDensityCentersX() {
//     return GPUSimulationHistoryInterface<T>::getDevDensityCentersX();
// }

// template <typename T>
// T* GPUSimulationHistory<T>::getDevDensityCentersY() {
//     return GPUSimulationHistoryInterface<T>::getDevDensityCentersY();
// }

// template <typename T>
// T* GPUSimulationHistory<T>::getDevDensityCentersZ() {
//     return GPUSimulationHistoryInterface<T>::getDevDensityCentersZ();
// }

template <typename T>
GPUSimulationHistory<T>::GPUSimulationHistory(int numIterations) : SimulationHistory<T>(numIterations) {
    this->allocateDeviceMemory(numIterations);
}

template <typename T>
void GPUSimulationHistory<T>::setNumIterations(int numIterations) {
    SimulationHistory<T>::setNumIterations(numIterations);
    this->allocateDeviceMemory(numIterations);
}

template <typename T>
void GPUSimulationHistory<T>::copyEnergiesFromDevice() {
    if (this->devEnergies == nullptr) throw std::runtime_error("Device memory not allocated");
    CUDA_CHECK(cudaMemcpy(this->energies.data(), this->devEnergies, this->energies.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void GPUSimulationHistory<T>::copyAngMomentumsFromDevice() {
    if (this->devAngMomentums == nullptr) throw std::runtime_error("Device memory not allocated");
    CUDA_CHECK(cudaMemcpy(this->angMomentums.data(), this->devAngMomentums, this->angMomentums.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void GPUSimulationHistory<T>::copyDensityCentersFromDevice() {
    if (this->devDensityCentersX == nullptr) throw std::runtime_error("Device memory not allocated");
    // Copy X, Y, Z components separately from device to host
    for (size_t i = 0; i < this->densityCenters.size(); ++i) {
        CUDA_CHECK(cudaMemcpy(&this->densityCenters[i][0], &this->devDensityCentersX[i], sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&this->densityCenters[i][1], &this->devDensityCentersY[i], sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&this->densityCenters[i][2], &this->devDensityCentersZ[i], sizeof(T), cudaMemcpyDeviceToHost));
    }
}

template <typename T>
void GPUSimulationHistory<T>::copyEnergiesToDevice() {
    if (this->devEnergies == nullptr) throw std::runtime_error("Device memory not allocated");
    CUDA_CHECK(cudaMemcpy(this->devEnergies, this->energies.data(), this->energies.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void GPUSimulationHistory<T>::copyAngMomentumsToDevice() {
    if (this->devAngMomentums == nullptr) throw std::runtime_error("Device memory not allocated");
    CUDA_CHECK(cudaMemcpy(this->devAngMomentums, this->angMomentums.data(), this->angMomentums.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void GPUSimulationHistory<T>::copyDensityCentersToDevice() {
    if (this->devDensityCentersX == nullptr) throw std::runtime_error("Device memory not allocated");
    // Copy X, Y, Z components separately from device to host
    for (size_t i = 0; i < this->densityCenters.size(); ++i) {
        CUDA_CHECK(cudaMemcpy(&this->devDensityCentersX[i], &this->densityCenters[i][0], sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&this->devDensityCentersY[i], &this->densityCenters[i][1], sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&this->devDensityCentersZ[i], &this->densityCenters[i][2], sizeof(T), cudaMemcpyHostToDevice));
    }
}

template <typename T>
void GPUSimulationHistory<T>::copyToDevice() {
    copyEnergiesToDevice();
    copyAngMomentumsToDevice();
    copyDensityCentersToDevice();
}

template <typename T>
void GPUSimulationHistory<T>::copyFromDevice() {
    copyEnergiesFromDevice();
    copyAngMomentumsFromDevice();
    copyDensityCentersFromDevice();
}

template <typename T>
GPUSimulationHistory<T>::~GPUSimulationHistory() {
}

// =========================== GPUMultiGalaxySimulationHistory ===========================
template<typename T, int numGalaxies, bool noIndData>
GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::GPUMultiGalaxySimulationHistory(int numIterations)
    : MultiGalaxySimulationHistory<T, numGalaxies, noIndData>(numIterations), GPUGalaxySimulationHistory<T>(numIterations)
{
    this->allocateDeviceMemory(numIterations);
    if constexpr (noIndData == false) {
        for (int i = 0; i < numGalaxies; i++) {
            this->devGalaxies[i].setNumIterations(numIterations);
        }
    }
}

template<typename T, int numGalaxies, bool noIndData>
void GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::setNumIterations(int numIterations) {
    GPUSimulationHistory<T>::setNumIterations(numIterations);
    // Copy MultiGalaxySimulationHistory::setNumIterations content to avoid double call
    if constexpr (noIndData == false) {
        for (int i = 0; i < numGalaxies; i++) {
            this->devGalaxies[i].setNumIterations(numIterations);
        }
    }
}

template<typename T, int numGalaxies, bool noIndData>
int GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::getNumIterations() const {
    return GPUGalaxySimulationHistory<T>::getNumIterations();
}

template<typename T, int numGalaxies, bool noIndData>
T GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::getEnergyAt(int iteration) const {
    return GPUGalaxySimulationHistory<T>::getEnergyAt(iteration);
}

template<typename T, int numGalaxies, bool noIndData>
void GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::setEnergyAt(int iteration, T energy) {
    GPUGalaxySimulationHistory<T>::setEnergyAt(iteration, energy);
}

template<typename T, int numGalaxies, bool noIndData>
const std::vector<T>& GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::getAllEnergy() const {
    return GPUGalaxySimulationHistory<T>::getAllEnergy();
}

template<typename T, int numGalaxies, bool noIndData>
std::vector<T>& GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::getAllEnergy() {
    return GPUGalaxySimulationHistory<T>::getAllEnergy();
}

template<typename T, int numGalaxies, bool noIndData>
void GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::setAllEnergy(std::vector<T> energies) {
    GPUGalaxySimulationHistory<T>::setAllEnergy(energies);
}

template<typename T, int numGalaxies, bool noIndData>
T GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::getAngMomentumAt(int iteration) const {
    return GPUGalaxySimulationHistory<T>::getAngMomentumAt(iteration);
}

template<typename T, int numGalaxies, bool noIndData>
void GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::setAngMomentumAt(int iteration, T angMomentum) {
    GPUGalaxySimulationHistory<T>::setAngMomentumAt(iteration, angMomentum);
}

template<typename T, int numGalaxies, bool noIndData>
const std::vector<T>& GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::getAllAngMomentum() const {
    return GPUGalaxySimulationHistory<T>::getAllAngMomentum();
}

template<typename T, int numGalaxies, bool noIndData>
std::vector<T>& GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::getAllAngMomentum() {
    return GPUGalaxySimulationHistory<T>::getAllAngMomentum();
}

template<typename T, int numGalaxies, bool noIndData>
void GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::setAllAngMomentum(std::vector<T> angMomentum) {
    GPUGalaxySimulationHistory<T>::setAllAngMomentum(angMomentum);
}

template<typename T, int numGalaxies, bool noIndData>
const std::array<T,3>& GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::getDensityCenterAt(int iteration) const {
    return GPUGalaxySimulationHistory<T>::getDensityCenterAt(iteration);
}

template<typename T, int numGalaxies, bool noIndData>
void GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::setDensityCenterAt(int iteration, const std::array<T,3>& densityCenter) {
    GPUGalaxySimulationHistory<T>::setDensityCenterAt(iteration, densityCenter);
}

template<typename T, int numGalaxies, bool noIndData>
const std::vector<std::array<T,3>>& GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::getAllDensityCenter() const {
    return GPUGalaxySimulationHistory<T>::getAllDensityCenter();
}

template<typename T, int numGalaxies, bool noIndData>
void GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::setAllDensityCenter(const std::vector<std::array<T,3>>& densityCenter) {
    GPUGalaxySimulationHistory<T>::setAllDensityCenter(densityCenter);
}

template<typename T, int numGalaxies, bool noIndData>
const GPUGalaxySimulationHistory<T>& 
const GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::getDevGalaxy(int i) const {
    return this->devGalaxies[i];
}

template<typename T, int numGalaxies, bool noIndData>
GPUGalaxySimulationHistory<T>& 
GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::getDevGalaxy(int i) {
    return this->devGalaxies[i];
}

template<typename T, int numGalaxies, bool noIndData>
const GalaxySimulationHistory<T>& 
GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::getGalaxy(int i) const {
    return this->devGalaxies[i];
}

template<typename T, int numGalaxies, bool noIndData>
GalaxySimulationHistory<T>& GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::getGalaxy(int i) {
    return this->devGalaxies[i];
}

template<typename T, int numGalaxies, bool noIndData>
void GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::updateGlobalProperties() {
    // This method should be called after copyFromDevice
    // Aggregate global properties from devGalaxies (local CPU copy)
    for (int i = 0; i < numGalaxies; i++) {
        const std::vector<T>& galaxyEnergies = this->devGalaxies[i].getAllEnergy();
        const std::vector<T>& galaxyAngMomentums = this->devGalaxies[i].getAllAngMomentum();
        const std::vector<std::array<T,3>>& galaxyDensityCenters = this->devGalaxies[i].getAllDensityCenter();
        // Summing energies
        std::transform(galaxyEnergies.begin(), galaxyEnergies.end(),
                        GPUGalaxySimulationHistory<T>::energies.begin(),
                        GPUGalaxySimulationHistory<T>::energies.begin(),
                        std::plus<T>());
        // sum_in_place<T>(GPUGalaxySimulationHistory<T>::energies, galaxyEnergies);
        
        // Summing momentums
        std::transform(galaxyAngMomentums.begin(), galaxyAngMomentums.end(),
                        GPUGalaxySimulationHistory<T>::angMomentums.begin(),
                        GPUGalaxySimulationHistory<T>::angMomentums.begin(),
                        std::plus<T>());
        // sum_in_place<T>(GPUGalaxySimulationHistory<T>::angMomentums, galaxyAngMomentums);
        
        // Summing density centers
        std::transform(galaxyDensityCenters.begin(), galaxyDensityCenters.end(),
                        GPUGalaxySimulationHistory<T>::densityCenters.begin(),
                        GPUGalaxySimulationHistory<T>::densityCenters.begin(),
                        [](const std::array<T,3>& x, const std::array<T,3>& y) {
                            return std::array<T,3>{
                                x[0] + y[0],
                                x[1] + y[1],
                                x[2] + y[2]
                            };
                        });
        // sum_in_place<T>(GPUGalaxySimulationHistory<T>::densityCenters, galaxyDensityCenters);
    }
}

template<typename T, int numGalaxies, bool noIndData>
void GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::copyToDevice() {
    GPUSimulationHistory<T>::copyToDevice();
    if constexpr (noIndData == false) {
        for (int i = 0; i < numGalaxies; i++) {
            this->devGalaxies[i].copyToDevice();
        }
    }
}

template<typename T, int numGalaxies, bool noIndData>
void GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::copyFromDevice() {
    GPUSimulationHistory<T>::copyFromDevice();
    if constexpr (noIndData == false) {
        for (int i = 0; i < numGalaxies; i++) {
            this->devGalaxies[i].copyFromDevice();
        }
    }
}

template<typename T, int numGalaxies, bool noIndData>
GPUMultiGalaxySimulationHistory<T, numGalaxies, noIndData>::~GPUMultiGalaxySimulationHistory() {
}

// Explicit template instantiation
template class GPUSimulationHistoryInterface<float>;
template class GPUSimulationHistory<float>;
template class GPUMultiGalaxySimulationHistory<float, 2, false>;

template class GPUSimulationHistoryInterface<double>;
template class GPUSimulationHistory<double>;
template class GPUMultiGalaxySimulationHistory<double, 2, false>;