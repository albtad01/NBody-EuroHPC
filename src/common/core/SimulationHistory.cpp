#include "SimulationHistory.hpp"


// ================================= SimulationHistory ==================================
template <typename T>
SimulationHistory<T>::SimulationHistory(int numIterations) 
{
    this->energies.resize(numIterations);
    this->angMomentums.resize(numIterations);
    this->densityCenters.resize(numIterations);
}

template <typename T>
void SimulationHistory<T>::setNumIterations(int numIterations) {
    this->energies.resize(numIterations);
    this->angMomentums.resize(numIterations);
    this->densityCenters.resize(numIterations);
}

template <typename T>
int SimulationHistory<T>::getNumIterations() const {
    return this->energies.size();
}

// === Energy
template <typename T>
T SimulationHistory<T>::getEnergyAt(int iteration) const {
    return this->energies[iteration];
}

template <typename T>
void SimulationHistory<T>::setEnergyAt(int iteration, T energy) {
    this->energies[iteration] = energy;
}

template <typename T>
const std::vector<T>& SimulationHistory<T>::getAllEnergy() const {
    return this->energies;
}

template <typename T>
std::vector<T>& SimulationHistory<T>::getAllEnergy() {
    return this->energies;
}

template <typename T>
void SimulationHistory<T>::setAllEnergy(std::vector<T> energies) {
    this->energies = energies;
}

// === Angular momentum
template <typename T>
T SimulationHistory<T>::getAngMomentumAt(int iteration) const {
    return this->angMomentums[iteration];
}

template <typename T>
void SimulationHistory<T>::setAngMomentumAt(int iteration, T angMomentum) {
    this->angMomentums[iteration] = angMomentum;
}

template <typename T>
const std::vector<T>& SimulationHistory<T>::getAllAngMomentum() const {
    return this->angMomentums;
}

template <typename T>
std::vector<T>& SimulationHistory<T>::getAllAngMomentum() {
    return this->angMomentums;
}

template <typename T>
void SimulationHistory<T>::setAllAngMomentum(std::vector<T> angMomentums) {
    this->angMomentums = angMomentums;
}

// === Density center
template <typename T>
const std::array<T,3>& SimulationHistory<T>::getDensityCenterAt(int iteration) const {
    return this->densityCenters[iteration];
}

template <typename T>
void SimulationHistory<T>::setDensityCenterAt(int iteration, const std::array<T,3>& densityCenter) {
    this->densityCenters[iteration] = densityCenter;
}

template <typename T>
const std::vector<std::array<T,3>>& SimulationHistory<T>::getAllDensityCenter() const {
    return this->densityCenters;
}

template <typename T>
void SimulationHistory<T>::setAllDensityCenter(const std::vector<std::array<T,3>>& densityCenter) {
    this->densityCenters = densityCenter;
}

// ============================ MultiGalaxySimulationHistory =============================
template<typename T, int numGalaxies, bool noIndData>
MultiGalaxySimulationHistory<T, numGalaxies, noIndData>::MultiGalaxySimulationHistory() 
    : GalaxySimulationHistory<T>()
{}

template<typename T, int numGalaxies, bool noIndData>
MultiGalaxySimulationHistory<T,numGalaxies, noIndData>::MultiGalaxySimulationHistory(int numIterations) 
    : GalaxySimulationHistory<T>(numIterations)
{
    if constexpr ( noIndData == false ) {
        //#pragma loop unroll
        for (int i = 0; i < numGalaxies; i++) {
            this->galaxies[i].setNumIterations(numIterations);
        }
    }
}

template<typename T, int numGalaxies, bool noIndData>
const GalaxySimulationHistory<T>& MultiGalaxySimulationHistory<T,numGalaxies, noIndData>::getGalaxy(int i) const {
    return this->galaxies[i];
}

template<typename T, int numGalaxies, bool noIndData>
GalaxySimulationHistory<T>& MultiGalaxySimulationHistory<T,numGalaxies, noIndData>::getGalaxy(int i){
    return this->galaxies[i];
}

template<typename T, int numGalaxies, bool noIndData>
void MultiGalaxySimulationHistory<T,numGalaxies, noIndData>::updateGlobalProperties() {
    //#pragma loop unroll
    for (int i = 0; i < numGalaxies; i++) {
        const std::vector<T>& galaxyEnergies = this->galaxies[i].getAllEnergy();
        const std::vector<T>& galaxyAngMomentums = this->galaxies[i].getAllAngMomentum();
        const std::vector<std::array<T,3>>& galaxyDensityCenters = this->galaxies[i].getAllDensityCenter();
        
        // Summing energies
        std::transform(galaxyEnergies.begin(), galaxyEnergies.end(),
                        this->energies.begin(),
                        this->energies.begin(),
                        std::plus<T>());
        
        // Summing momentums
        std::transform(galaxyAngMomentums.begin(), galaxyAngMomentums.end(),
                        this->angMomentums.begin(),
                        this->angMomentums.begin(),   // in-place
                        std::plus<T>());
        
        // Summing density centers
        std::transform(galaxyDensityCenters.begin(), galaxyDensityCenters.end(),
                        this->densityCenters.begin(),
                        this->densityCenters.begin(),   // in-place
                        [](const std::array<T,3>& x, const std::array<T,3>& y) {
                            return std::array<T,3>{
                                x[0] + y[0],
                                x[1] + y[1],
                                x[2] + y[2]
                            };
                        });
    }
}

template class SimulationHistory<float>;
template class MultiGalaxySimulationHistory<float,2,false>;