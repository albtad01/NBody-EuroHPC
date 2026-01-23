#ifndef GPU_SIMULATION_HISTORY_HPP
#define GPU_SIMULATION_HISTORY_HPP

#include "SimulationHistory.hpp"
#include <cuda_runtime.h>

template <typename T>
class GPUSimulationHistoryInterface {
    protected:
        T* devEnergies = nullptr;
        T* devAngMomentums = nullptr;
        T* devDensityCentersX = nullptr;
        T* devDensityCentersY = nullptr;
        T* devDensityCentersZ = nullptr;
        int allocatedIterations = 0;
    public:
        GPUSimulationHistoryInterface() = default;

        // === Energy
        virtual T* getDevEnergy();

        // === Angular Momentum
        virtual T* getDevAngMomentum();

        // === Density Centers
        virtual T* getDevDensityCentersX();
        virtual T* getDevDensityCentersY();
        virtual T* getDevDensityCentersZ();

        // === Memory management
        virtual void allocateDeviceMemory(int numIterations);
        virtual void deallocateDeviceMemory();

        virtual ~GPUSimulationHistoryInterface();
};

template <typename T>
class GPUSimulationHistory : public SimulationHistory<T>, public GPUSimulationHistoryInterface<T> {
    public:
        GPUSimulationHistory() = default;
        GPUSimulationHistory(int numIterations);

        virtual void setNumIterations(int numIterations) override;
        
        // === Copy methods (Host to Device)
        virtual void copyEnergiesToDevice();
        virtual void copyAngMomentumsToDevice();
        virtual void copyDensityCentersToDevice();
        virtual void copyToDevice();

        // === Copy methods (Device to Host)
        virtual void copyEnergiesFromDevice();
        virtual void copyAngMomentumsFromDevice();
        virtual void copyDensityCentersFromDevice();
        virtual void copyFromDevice();

        virtual ~GPUSimulationHistory();
};

template<typename T>
using GPUGalaxySimulationHistory = GPUSimulationHistory<T>;

template<typename T, int numGalaxies, bool noGalaxyIndividualData = false>
class GPUMultiGalaxySimulationHistory : 
        public MultiGalaxySimulationHistory<T, numGalaxies, noGalaxyIndividualData>, 
        public GPUGalaxySimulationHistory<T> {
    private:
        std::array<GPUGalaxySimulationHistory<T>, numGalaxies> devGalaxies;
    public:
        GPUMultiGalaxySimulationHistory() = default;
        GPUMultiGalaxySimulationHistory(int numIterations);

        // === Disambiguate all SimulationHistory methods ===
        virtual void setNumIterations(int numIterations) override;
        virtual int getNumIterations() const override;

        // === Energy
        virtual T getEnergyAt(int iteration) const override;
        virtual void setEnergyAt(int iteration, T energy) override;
        virtual const std::vector<T>& getAllEnergy() const override;
        virtual std::vector<T>& getAllEnergy() override;
        virtual void setAllEnergy(std::vector<T> energies) override;

        // === Angular Momentum
        virtual T getAngMomentumAt(int iteration) const override;
        virtual void setAngMomentumAt(int iteration, T angMomentum) override;
        virtual const std::vector<T>& getAllAngMomentum() const override;
        virtual std::vector<T>& getAllAngMomentum() override;
        virtual void setAllAngMomentum(std::vector<T> angMomentum) override;

        // === Density Center
        virtual const std::array<T,3>& getDensityCenterAt(int iteration) const override;
        virtual void setDensityCenterAt(int iteration, const std::array<T,3>& densityCenter) override;
        virtual const std::vector<std::array<T,3>>& getAllDensityCenter() const override;
        virtual void setAllDensityCenter(const std::vector<std::array<T,3>>& densityCenter) override;

        // === GPU-specific methods
        virtual const GPUGalaxySimulationHistory<T>& getDevGalaxy(int i) const;
        virtual GPUGalaxySimulationHistory<T>& getDevGalaxy(int i);
        
        // Override inherited multi-galaxy methods to use devGalaxies instead of galaxies
        virtual const GalaxySimulationHistory<T>& getGalaxy(int i) const override;
        virtual GalaxySimulationHistory<T>& getGalaxy(int i) override;
        virtual void updateGlobalProperties() override;

        virtual void copyToDevice();
        virtual void copyFromDevice();

        virtual ~GPUMultiGalaxySimulationHistory();
};

#endif