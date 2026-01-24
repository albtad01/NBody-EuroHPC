#ifndef SIMULATION_HISTORY_HPP
#define SIMULATION_HISTORY_HPP


#include <vector>
#include <array>
#include <algorithm>
#include <string>

template <typename T>
class SimulationHistory {
    protected:
        std::vector<T> energies;
        std::vector<T> angMomentums;
        std::vector<std::array<T,3>> densityCenters;
    public:
        SimulationHistory() = default;
        SimulationHistory(int numIterations);

        virtual void setNumIterations(int numIterations);
        virtual int getNumIterations() const;

        // === Energy
        virtual T getEnergyAt(int iteration) const;
        virtual void setEnergyAt(int iteration, T energy);

        virtual const std::vector<T>& getAllEnergy() const;
        virtual std::vector<T>& getAllEnergy();
        virtual void setAllEnergy(std::vector<T> energies);

        // === Angular Momentum
        virtual T getAngMomentumAt(int iteration) const;
        virtual void setAngMomentumAt(int iteration, T angMomentum);
        virtual const std::vector<T>& getAllAngMomentum() const;
        virtual std::vector<T>& getAllAngMomentum();
        virtual void setAllAngMomentum(std::vector<T> angMomentum);

        // === Density Center
        virtual const std::array<T,3>& getDensityCenterAt(int iteration) const;
        virtual void setDensityCenterAt(int iteration, const std::array<T,3>& densityCenter);
        virtual const std::vector<std::array<T,3>>& getAllDensityCenter() const;
        virtual void setAllDensityCenter(const std::vector<std::array<T,3>>& densityCenter);

        // === Export
        // CSV columns: iteration,energy,ang_momentum,density_center_x,density_center_y,density_center_z
        virtual void saveMetricsToCSV(const std::string& filePath) const;

        virtual ~SimulationHistory() = default;
};

// Alias per compatibilità
template<typename T>
using GalaxySimulationHistory = SimulationHistory<T>;


template<typename T, int numGalaxies, bool noGalaxyIndividualData = false>
class MultiGalaxySimulationHistory : public GalaxySimulationHistory<T> {
    private:
        std::array<GalaxySimulationHistory<T>, numGalaxies> galaxies;
    public:
        // =============== Main methods inherited from SimulationHistory ================
        // virtual T getEnergyAt(int iteration) override;
        // virtual const std::vector<T> getAllEnergy() override;

        // virtual T getAngMomentumAt(int iteration) override;
        // virtual const std::vector<T> getAllAngMomentum() override;

        // ================ Methods inherited from GalaxySimulationHistory ==============
        // virtual const std::array<T,3> getDensityCenterAt(int iteration) override;
        // virtual const std::vector<std::array<T,3>> getAllDensityCenter() override;

        MultiGalaxySimulationHistory();
        MultiGalaxySimulationHistory(int numIterations);

        virtual const GalaxySimulationHistory<T>& getGalaxy(int i) const;
        virtual GalaxySimulationHistory<T>& getGalaxy(int i);
        virtual void updateGlobalProperties();

        virtual ~MultiGalaxySimulationHistory() = default;
};
#endif