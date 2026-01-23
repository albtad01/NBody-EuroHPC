#include <catch.hpp>
#include <array>
#include <vector>
#include <cmath>
#include <numeric>

#include "core/SimulationHistory.hpp"
#include "core/SimulationHistoryGPU.hpp"

// ======================== CPU Tests ========================

TEST_CASE("SimulationHistory basic functionality", "[SimulationHistory]") {
    const int numIterations = 10;
    SimulationHistory<float> history(numIterations);
    
    SECTION("Initialization and getNumIterations") {
        REQUIRE(history.getNumIterations() == numIterations);
    }
    
    SECTION("Energy setters and getters") {
        float energyValue = 42.5f;
        history.setEnergyAt(0, energyValue);
        REQUIRE(history.getEnergyAt(0) == energyValue);
    }
    
    SECTION("Angular Momentum setters and getters") {
        float angMomValue = 3.14f;
        history.setAngMomentumAt(0, angMomValue);
        REQUIRE(history.getAngMomentumAt(0) == angMomValue);
    }
    
    SECTION("Density Center setters and getters") {
        std::array<float, 3> densityCenter = {1.0f, 2.0f, 3.0f};
        history.setDensityCenterAt(0, densityCenter);
        auto retrieved = history.getDensityCenterAt(0);
        REQUIRE(retrieved[0] == densityCenter[0]);
        REQUIRE(retrieved[1] == densityCenter[1]);
        REQUIRE(retrieved[2] == densityCenter[2]);
    }
    
    SECTION("setNumIterations resizes correctly") {
        int newSize = 20;
        history.setNumIterations(newSize);
        REQUIRE(history.getNumIterations() == newSize);
    }
}

TEST_CASE("SimulationHistory bulk operations", "[SimulationHistory]") {
    const int numIterations = 5;
    SimulationHistory<float> history(numIterations);
    
    SECTION("setAllEnergy and getAllEnergy") {
        std::vector<float> energies = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        history.setAllEnergy(energies);
        auto retrieved = history.getAllEnergy();
        REQUIRE(retrieved == energies);
    }
    
    SECTION("setAllAngMomentum and getAllAngMomentum") {
        std::vector<float> angMomentums = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        history.setAllAngMomentum(angMomentums);
        auto retrieved = history.getAllAngMomentum();
        REQUIRE(retrieved == angMomentums);
    }
    
    SECTION("setAllDensityCenter and getAllDensityCenter") {
        std::vector<std::array<float, 3>> densityCenters = {
            {{1.0f, 2.0f, 3.0f}},
            {{4.0f, 5.0f, 6.0f}},
            {{7.0f, 8.0f, 9.0f}},
            {{10.0f, 11.0f, 12.0f}},
            {{13.0f, 14.0f, 15.0f}}
        };
        history.setAllDensityCenter(densityCenters);
        auto retrieved = history.getAllDensityCenter();
        REQUIRE(retrieved == densityCenters);
    }
}


TEST_CASE("MultiGalaxySimulationHistory", "[MultiGalaxySimulationHistory]") {
    const int numIterations = 5;
    const int numGalaxies = 2;
    MultiGalaxySimulationHistory<float, numGalaxies, false> multiGalaxy(numIterations);
    
    SECTION("Initialization") {
        REQUIRE(multiGalaxy.getNumIterations() == numIterations);
    }
    
    SECTION("Access individual galaxies") {
        auto& galaxy0 = multiGalaxy.getGalaxy(0);
        auto& galaxy1 = multiGalaxy.getGalaxy(1);
        REQUIRE(galaxy0.getNumIterations() == numIterations);
        REQUIRE(galaxy1.getNumIterations() == numIterations);
    }
    
    SECTION("Set individual galaxy data") {
        auto& galaxy0 = multiGalaxy.getGalaxy(0);
        std::vector<float> energies = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        galaxy0.setAllEnergy(energies);
        REQUIRE(galaxy0.getAllEnergy() == energies);
    }
    
    SECTION("updateGlobalProperties aggregates correctly") {
        // Set energies for galaxy 0
        auto& galaxy0 = multiGalaxy.getGalaxy(0);
        std::vector<float> energies0 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        galaxy0.setAllEnergy(energies0);
        
        // Set energies for galaxy 1
        auto& galaxy1 = multiGalaxy.getGalaxy(1);
        std::vector<float> energies1 = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        galaxy1.setAllEnergy(energies1);
        
        // Call updateGlobalProperties
        multiGalaxy.updateGlobalProperties();
        
        // Check that global energies are sum of both galaxies
        auto globalEnergies = multiGalaxy.getAllEnergy();
        std::vector<float> expected = {6.0f, 6.0f, 6.0f, 6.0f, 6.0f};
        for (int i = 0; i < numIterations; i++) {
            REQUIRE_THAT(globalEnergies[i], Catch::Matchers::WithinRel(expected[i]));
        }
    }
}

// ======================== GPU Tests ========================

TEST_CASE("GPUSimulationHistory memory management", "[GPUSimulationHistory]") {
    const int numIterations = 10;
    GPUSimulationHistory<float> gpuHistory(numIterations);
    
    SECTION("Device memory is allocated") {
        REQUIRE(gpuHistory.getDevEnergy() != nullptr);
        REQUIRE(gpuHistory.getDevAngMomentum() != nullptr);
        REQUIRE(gpuHistory.getDevDensityCentersX() != nullptr);
        REQUIRE(gpuHistory.getDevDensityCentersY() != nullptr);
        REQUIRE(gpuHistory.getDevDensityCentersZ() != nullptr);
    }
    
    SECTION("setNumIterations reallocates device memory") {
        int newSize = 20;
        gpuHistory.setNumIterations(newSize);
        REQUIRE(gpuHistory.getNumIterations() == newSize);
        REQUIRE(gpuHistory.getDevEnergy() != nullptr);
    }
}

TEST_CASE("GPUSimulationHistory copy operations", "[GPUSimulationHistory]") {
    const int numIterations = 5;
    GPUSimulationHistory<float> gpuHistory(numIterations);
    
    SECTION("copyToDevice and copyFromDevice consistency") {
        // Set some host data
        std::vector<float> hostEnergies = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        gpuHistory.setAllEnergy(hostEnergies);
        
        // Copy to device and back
        gpuHistory.copyToDevice();
        gpuHistory.copyFromDevice();
        
        // Verify data is preserved
        auto retrieved = gpuHistory.getAllEnergy();
        for (int i = 0; i < numIterations; i++) {
            CAPTURE(i);
            REQUIRE_THAT(retrieved[i], Catch::Matchers::WithinRel(hostEnergies[i]));
            
        }
    }
    
    SECTION("copyEnergiesToDevice preserves data") {
        std::vector<float> energies = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
        gpuHistory.setAllEnergy(energies);
        gpuHistory.copyEnergiesToDevice();
        gpuHistory.copyFromDevice();
        
        auto retrieved = gpuHistory.getAllEnergy();
        for (int i = 0; i < numIterations; i++) {
            REQUIRE_THAT(retrieved[i], Catch::Matchers::WithinRel(energies[i]));
        }
    }
    
    SECTION("copyAngMomentumsToDevice preserves data") {
        std::vector<float> angMomentums = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        gpuHistory.setAllAngMomentum(angMomentums);
        gpuHistory.copyAngMomentumsToDevice();
        gpuHistory.copyFromDevice();
        
        auto retrieved = gpuHistory.getAllAngMomentum();
        for (int i = 0; i < numIterations; i++) {
            REQUIRE_THAT(retrieved[i], Catch::Matchers::WithinRel(angMomentums[i]));
        }
    }
    
    SECTION("copyDensityCentersToDevice preserves data") {
        std::vector<std::array<float, 3>> densityCenters = {
            {{1.0f, 2.0f, 3.0f}},
            {{4.0f, 5.0f, 6.0f}},
            {{7.0f, 8.0f, 9.0f}},
            {{10.0f, 11.0f, 12.0f}},
            {{13.0f, 14.0f, 15.0f}}
        };
        gpuHistory.setAllDensityCenter(densityCenters);
        gpuHistory.copyDensityCentersToDevice();
        gpuHistory.copyFromDevice();
        
        auto retrieved = gpuHistory.getAllDensityCenter();
        for (int i = 0; i < numIterations; i++) {
            REQUIRE(retrieved[i][0] == densityCenters[i][0]);
            REQUIRE(retrieved[i][1] == densityCenters[i][1]);
            REQUIRE(retrieved[i][2] == densityCenters[i][2]);
        }
    }
}

#include <iostream>
TEST_CASE("GPUMultiGalaxySimulationHistory", "[GPUMultiGalaxySimulationHistory]") {
    const int numIterations = 5;
    const int numGalaxies = 2;
    GPUMultiGalaxySimulationHistory<float, numGalaxies, false> gpuMultiGalaxy(numIterations);
    
    SECTION("Initialization") {
        REQUIRE(gpuMultiGalaxy.getNumIterations() == numIterations);
    }
    
    SECTION("Access devGalaxies through getDevGalaxy") {
        auto& devGalaxy0 = gpuMultiGalaxy.getDevGalaxy(0);
        auto& devGalaxy1 = gpuMultiGalaxy.getDevGalaxy(1);
        REQUIRE(devGalaxy0.getNumIterations() == numIterations);
        REQUIRE(devGalaxy1.getNumIterations() == numIterations);
    }
    
    SECTION("getGalaxy returns devGalaxies (not galaxies)") {
        auto& galaxy0 = gpuMultiGalaxy.getGalaxy(0);
        auto& galaxy1 = gpuMultiGalaxy.getGalaxy(1);
        REQUIRE(galaxy0.getNumIterations() == numIterations);
        REQUIRE(galaxy1.getNumIterations() == numIterations);
    }
    
    SECTION("Set individual devGalaxy data") {
        auto& devGalaxy0 = gpuMultiGalaxy.getDevGalaxy(0);
        std::vector<float> energies = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        devGalaxy0.setAllEnergy(energies);
    }
    
    SECTION("copyToDevice and copyFromDevice with devGalaxies") {
        auto& devGalaxy0 = gpuMultiGalaxy.getDevGalaxy(0);
        std::vector<float> energies0 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        devGalaxy0.setAllEnergy(energies0);
        
        auto& devGalaxy1 = gpuMultiGalaxy.getDevGalaxy(1);
        std::vector<float> energies1 = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        devGalaxy1.setAllEnergy(energies1);
        
        // Copy to and from device
        gpuMultiGalaxy.copyToDevice();
        gpuMultiGalaxy.copyFromDevice();
        
        // Verify data is preserved
        auto retrieved0 = gpuMultiGalaxy.getDevGalaxy(0).getAllEnergy();
        auto retrieved1 = gpuMultiGalaxy.getDevGalaxy(1).getAllEnergy();
        
        for (int i = 0; i < numIterations; i++) {
            REQUIRE_THAT(retrieved0[i], Catch::Matchers::WithinRel(energies0[i]));
            REQUIRE_THAT(retrieved1[i], Catch::Matchers::WithinRel(energies1[i]));
        }
    }

    SECTION("updateGlobalProperties aggregates from devGalaxies") {
        // Set energies for devGalaxy 0
        auto& devGalaxy0 = gpuMultiGalaxy.getDevGalaxy(0);
        std::vector<float> energies0 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        devGalaxy0.setAllEnergy(energies0);
        
        // Set energies for devGalaxy 1
        auto& devGalaxy1 = gpuMultiGalaxy.getDevGalaxy(1);
        std::vector<float> energies1 = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        devGalaxy1.setAllEnergy(energies1);
        REQUIRE(devGalaxy1.getAllEnergy() == energies1);
        
        
        // Call updateGlobalProperties
        gpuMultiGalaxy.updateGlobalProperties();
        
        // Check that global energies are sum of both devGalaxies
        auto globalEnergies = gpuMultiGalaxy.getAllEnergy();
        std::vector<float> expected = {6.0f, 6.0f, 6.0f, 6.0f, 6.0f};
        for (int i = 0; i < numIterations; i++) {
            REQUIRE_THAT(globalEnergies[i], Catch::Matchers::WithinRel(expected[i]));
        }
    }
    
    
    SECTION("setNumIterations updates devGalaxies") {
        int newSize = 10;
        gpuMultiGalaxy.setNumIterations(newSize);
        
        REQUIRE(gpuMultiGalaxy.getNumIterations() == newSize);
        REQUIRE(gpuMultiGalaxy.getDevGalaxy(0).getNumIterations() == newSize);
        REQUIRE(gpuMultiGalaxy.getDevGalaxy(1).getNumIterations() == newSize);
    }
}

TEST_CASE("GPUMultiGalaxySimulationHistory density centers aggregation", "[GPUMultiGalaxySimulationHistory]") {
    const int numIterations = 3;
    const int numGalaxies = 2;
    GPUMultiGalaxySimulationHistory<float, numGalaxies, false> gpuMultiGalaxy(numIterations);
    
    SECTION("updateGlobalProperties aggregates density centers") {
        auto& devGalaxy0 = gpuMultiGalaxy.getDevGalaxy(0);
        std::vector<std::array<float, 3>> centers0 = {
            {{1.0f, 1.0f, 1.0f}},
            {{2.0f, 2.0f, 2.0f}},
            {{3.0f, 3.0f, 3.0f}}
        };
        devGalaxy0.setAllDensityCenter(centers0);
        
        auto& devGalaxy1 = gpuMultiGalaxy.getDevGalaxy(1);
        std::vector<std::array<float, 3>> centers1 = {
            {{1.0f, 1.0f, 1.0f}},
            {{2.0f, 2.0f, 2.0f}},
            {{3.0f, 3.0f, 3.0f}}
        };
        devGalaxy1.setAllDensityCenter(centers1);
        
        gpuMultiGalaxy.updateGlobalProperties();
        
        auto globalCenters = gpuMultiGalaxy.getAllDensityCenter();
        for (int i = 0; i < numIterations; i++) {
            REQUIRE_THAT(globalCenters[i][0], Catch::Matchers::WithinRel(2.0*(i+1)));
            REQUIRE_THAT(globalCenters[i][1], Catch::Matchers::WithinRel(2.0*(i+1)));
            REQUIRE_THAT(globalCenters[i][2], Catch::Matchers::WithinRel(2.0*(i+1)));
        }
    }
}
// ======================== CUDA Kernel Tests ========================

// Kernel to multiply all energies by a scalar
template <typename T>
__global__ void multiplyEnergiesKernel(T* energies, int numIterations, T factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numIterations) {
        energies[idx] *= factor;
    }
}

// Kernel to normalize angular momentums
template <typename T>
__global__ void normalizeAngMomentumsKernel(T* angMomentums, int numIterations, T maxValue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numIterations) {
        angMomentums[idx] /= maxValue;
    }
}

// Kernel to transform density centers (translate by offset)
__global__ void translateDensityCentersKernel(float* centersX, float* centersY, float* centersZ, 
                                               int numIterations, float offsetX, float offsetY, float offsetZ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numIterations) {
        centersX[idx] += offsetX;
        centersY[idx] += offsetY;
        centersZ[idx] += offsetZ;
    }
}

// Kernel to compute sum of energies
__global__ void sumEnergiesKernel(float* energies, int numIterations, float* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float sharedData[256];
    
    sharedData[threadIdx.x] = (idx < numIterations) ? energies[idx] : 0.0f;
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedData[threadIdx.x] += sharedData[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(result, sharedData[0]);
    }
}

TEST_CASE("GPU kernel: multiply energies", "[GPUKernel][Energy]") {
    const int numIterations = 10;
    GPUSimulationHistory<float> gpuHistory(numIterations);
    
    SECTION("Multiply energies on GPU by scalar factor") {
        // Set up CPU data
        std::vector<float> cpuEnergies(numIterations);
        for (int i = 0; i < numIterations; ++i) {
            cpuEnergies[i] = static_cast<float>(i + 1);
        }
        gpuHistory.setAllEnergy(cpuEnergies);
        gpuHistory.copyToDevice();
        
        // Apply kernel
        float factor = 2.5f;
        int blockSize = 256;
        int gridSize = (numIterations + blockSize - 1) / blockSize;
        multiplyEnergiesKernel<<<gridSize, blockSize>>>(gpuHistory.getDevEnergy(), numIterations, factor);
        cudaDeviceSynchronize();
        
        // Copy back from device
        gpuHistory.copyFromDevice();
        
        // Verify results
        auto resultEnergies = gpuHistory.getAllEnergy();
        for (int i = 0; i < numIterations; ++i) {
            REQUIRE_THAT(resultEnergies[i], Catch::Matchers::WithinRel((i + 1) * factor));
        }
    }
}

TEST_CASE("GPU kernel: normalize angular momentums", "[GPUKernel][AngularMomentum]") {
    const int numIterations = 8;
    GPUSimulationHistory<float> gpuHistory(numIterations);
    
    SECTION("Normalize angular momentums on GPU") {
        // Set up CPU data
        std::vector<float> cpuAngMom(numIterations);
        for (int i = 0; i < numIterations; ++i) {
            cpuAngMom[i] = 10.0f + static_cast<float>(i);
        }
        gpuHistory.setAllAngMomentum(cpuAngMom);
        gpuHistory.copyToDevice();
        
        // Apply kernel
        float maxValue = 15.0f;
        int blockSize = 256;
        int gridSize = (numIterations + blockSize - 1) / blockSize;
        normalizeAngMomentumsKernel<<<gridSize, blockSize>>>(gpuHistory.getDevAngMomentum(), numIterations, maxValue);
        cudaDeviceSynchronize();
        
        // Copy back from device
        gpuHistory.copyFromDevice();
        
        // Verify results
        auto resultAngMom = gpuHistory.getAllAngMomentum();
        for (int i = 0; i < numIterations; ++i) {
            float expected = (10.0f + static_cast<float>(i)) / maxValue;
            REQUIRE_THAT(resultAngMom[i], Catch::Matchers::WithinRel(expected, 1e-5f));
        }
    }
}

TEST_CASE("GPU kernel: translate density centers", "[GPUKernel][DensityCenter]") {
    const int numIterations = 5;
    GPUSimulationHistory<float> gpuHistory(numIterations);
    
    SECTION("Translate density centers on GPU") {
        // Set up CPU data
        std::vector<std::array<float, 3>> densityCenters(numIterations);
        for (int i = 0; i < numIterations; ++i) {
            densityCenters[i] = {{1.0f * i, 2.0f * i, 3.0f * i}};
        }
        gpuHistory.setAllDensityCenter(densityCenters);
        gpuHistory.copyToDevice();
        
        // Apply kernel
        float offsetX = 5.0f, offsetY = 10.0f, offsetZ = 15.0f;
        int blockSize = 256;
        int gridSize = (numIterations + blockSize - 1) / blockSize;
        translateDensityCentersKernel<<<gridSize, blockSize>>>(
            gpuHistory.getDevDensityCentersX(),
            gpuHistory.getDevDensityCentersY(),
            gpuHistory.getDevDensityCentersZ(),
            numIterations, offsetX, offsetY, offsetZ
        );
        cudaDeviceSynchronize();
        
        // Copy back from device
        gpuHistory.copyFromDevice();
        
        // Verify results
        auto resultCenters = gpuHistory.getAllDensityCenter();
        for (int i = 0; i < numIterations; ++i) {
            REQUIRE_THAT(resultCenters[i][0], Catch::Matchers::WithinRel(1.0f * i + offsetX));
            REQUIRE_THAT(resultCenters[i][1], Catch::Matchers::WithinRel(2.0f * i + offsetY));
            REQUIRE_THAT(resultCenters[i][2], Catch::Matchers::WithinRel(3.0f * i + offsetZ));
        }
    }
}

TEST_CASE("GPU kernel: sum energies with reduction", "[GPUKernel][Energy][Reduction]") {
    const int numIterations = 32;
    GPUSimulationHistory<float> gpuHistory(numIterations);
    
    SECTION("Compute sum of energies using GPU reduction") {
        // Set up CPU data
        std::vector<float> cpuEnergies(numIterations);
        float expectedSum = 0.0f;
        for (int i = 0; i < numIterations; ++i) {
            cpuEnergies[i] = static_cast<float>(i + 1);
            expectedSum += cpuEnergies[i];
        }
        gpuHistory.setAllEnergy(cpuEnergies);
        gpuHistory.copyToDevice();
        
        // Allocate device memory for result
        float* devResult = nullptr;
        cudaMalloc(&devResult, sizeof(float));
        float hostResult = 0.0f;
        cudaMemcpy(devResult, &hostResult, sizeof(float), cudaMemcpyHostToDevice);
        
        // Apply kernel
        int blockSize = 256;
        int gridSize = (numIterations + blockSize - 1) / blockSize;
        sumEnergiesKernel<<<gridSize, blockSize>>>(gpuHistory.getDevEnergy(), numIterations, devResult);
        cudaDeviceSynchronize();
        
        // Copy result back
        cudaMemcpy(&hostResult, devResult, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(devResult);
        
        // Verify result
        REQUIRE_THAT(hostResult, Catch::Matchers::WithinRel(expectedSum, 1e-5f));
    }
}

TEST_CASE("GPU kernel: multi-galaxy energy transformation", "[GPUKernel][MultiGalaxy]") {
    const int numIterations = 6;
    const int numGalaxies = 2;
    GPUMultiGalaxySimulationHistory<float, numGalaxies, false> gpuMultiGalaxy(numIterations);
    
    SECTION("Transform energies in all galaxies on GPU") {
        // Set up data for galaxy 0
        std::vector<float> energies0(numIterations);
        for (int i = 0; i < numIterations; ++i) {
            energies0[i] = 1.0f;
        }
        gpuMultiGalaxy.getDevGalaxy(0).setAllEnergy(energies0);
        gpuMultiGalaxy.getDevGalaxy(0).copyToDevice();
        
        // Set up data for galaxy 1
        std::vector<float> energies1(numIterations);
        for (int i = 0; i < numIterations; ++i) {
            energies1[i] = 2.0f;
        }
        gpuMultiGalaxy.getDevGalaxy(1).setAllEnergy(energies1);
        gpuMultiGalaxy.getDevGalaxy(1).copyToDevice();
        
        // Apply kernel to both galaxies
        float factor = 3.5f;
        int blockSize = 256;
        int gridSize = (numIterations + blockSize - 1) / blockSize;
        
        multiplyEnergiesKernel<<<gridSize, blockSize>>>(
            gpuMultiGalaxy.getDevGalaxy(0).getDevEnergy(), numIterations, factor);
        multiplyEnergiesKernel<<<gridSize, blockSize>>>(
            gpuMultiGalaxy.getDevGalaxy(1).getDevEnergy(), numIterations, factor);
        cudaDeviceSynchronize();
        
        // Copy back from device
        gpuMultiGalaxy.getDevGalaxy(0).copyFromDevice();
        gpuMultiGalaxy.getDevGalaxy(1).copyFromDevice();
        
        // Verify results for galaxy 0
        auto resultEnergies0 = gpuMultiGalaxy.getDevGalaxy(0).getAllEnergy();
        for (int i = 0; i < numIterations; ++i) {
            REQUIRE_THAT(resultEnergies0[i], Catch::Matchers::WithinRel(1.0f * factor));
        }
        
        // Verify results for galaxy 1
        auto resultEnergies1 = gpuMultiGalaxy.getDevGalaxy(1).getAllEnergy();
        for (int i = 0; i < numIterations; ++i) {
            REQUIRE_THAT(resultEnergies1[i], Catch::Matchers::WithinRel(2.0f * factor));
        }
    }
}

TEST_CASE("GPU kernel: combined density center operations", "[GPUKernel][DensityCenter][Combined]") {
    const int numIterations = 4;
    GPUSimulationHistory<float> gpuHistory(numIterations);
    
    SECTION("Apply multiple transformations to density centers") {
        // Set up initial CPU data
        std::vector<std::array<float, 3>> densityCenters(numIterations);
        for (int i = 0; i < numIterations; ++i) {
            densityCenters[i] = {{0.5f, 0.5f, 0.5f}};
        }
        gpuHistory.setAllDensityCenter(densityCenters);
        gpuHistory.copyToDevice();
        
        // Apply first transformation: translate
        float offset1X = 1.0f, offset1Y = 1.0f, offset1Z = 1.0f;
        int blockSize = 256;
        int gridSize = (numIterations + blockSize - 1) / blockSize;
        translateDensityCentersKernel<<<gridSize, blockSize>>>(
            gpuHistory.getDevDensityCentersX(),
            gpuHistory.getDevDensityCentersY(),
            gpuHistory.getDevDensityCentersZ(),
            numIterations, offset1X, offset1Y, offset1Z
        );
        cudaDeviceSynchronize();
        
        // Apply second transformation: translate again
        float offset2X = 0.5f, offset2Y = 0.5f, offset2Z = 0.5f;
        translateDensityCentersKernel<<<gridSize, blockSize>>>(
            gpuHistory.getDevDensityCentersX(),
            gpuHistory.getDevDensityCentersY(),
            gpuHistory.getDevDensityCentersZ(),
            numIterations, offset2X, offset2Y, offset2Z
        );
        cudaDeviceSynchronize();
        
        // Copy back from device
        gpuHistory.copyFromDevice();
        
        // Verify results
        auto resultCenters = gpuHistory.getAllDensityCenter();
        float expectedX = 0.5f + offset1X + offset2X;
        float expectedY = 0.5f + offset1Y + offset2Y;
        float expectedZ = 0.5f + offset1Z + offset2Z;
        
        for (int i = 0; i < numIterations; ++i) {
            REQUIRE_THAT(resultCenters[i][0], Catch::Matchers::WithinRel(expectedX));
            REQUIRE_THAT(resultCenters[i][1], Catch::Matchers::WithinRel(expectedY));
            REQUIRE_THAT(resultCenters[i][2], Catch::Matchers::WithinRel(expectedZ));
        }
    }
}