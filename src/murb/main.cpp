#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>

#include "ogl/SpheresVisu.hpp"
#include "ogl/SpheresVisuNo.hpp"
#ifdef VISU
#include "ogl/OGLSpheresVisuGS.hpp"
#include "ogl/OGLSpheresVisuInst.hpp"
#endif

#include "core/Bodies.hpp"
#include "utils/Perf.hpp"

#include "implem/SimulationNBodyNaive.hpp"
#include "implem/SimulationNBodyOpenMP.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "implem/SimulationNBodyCUDATileFullDevice.hpp"
#endif

/* global variables */
unsigned long NBodies;               /*!< Number of bodies. */
unsigned long NIterations;           /*!< Number of iterations. */
std::string ImplTag = "cpu+naive";   /*!< Implementation id. */
bool Verbose = false;                /*!< Mode verbose. */
bool GSEnable = true;                /*!< Enable geometry shader. */
bool VisuEnable = true;              /*!< Enable visualization. */
bool VisuColor = true;               /*!< Enable visualization with colors. */
float Dt = 3600;                     /*!< Time step in seconds. */
float MinDt = 200;                   /*!< Minimum time step. */
float Softening = 2e+08;             /*!< Softening factor value. */
unsigned int WinWidth = 1024;        /*!< Window width for visualization. */
unsigned int WinHeight = 768;        /*!< Window height for visualization. */
unsigned int LocalWGSize = 32;       /*!< OpenCL local workgroup size. */
std::string BodiesScheme = "galaxy"; /*!< Initial condition of the bodies. */
bool ShowGFlops = false;             /*!< Display the GFlop/s. */

void printUsage() {
    std::cout << "Usage: murb -n BODIES -i ITERATIONS [--im BACKEND] [--nv] [--gf] [--dt SECONDS]\n"
              << "Phase 1 backends: cpu+naive, cpu+omp, gpu+tile+full (CUDA build only).\n"
              << "Options: --visu (local OpenGL), --scheme galaxy|random, -v, --help.\n"
              << "Recording, bin+player and other backends are deferred.\n";
}

unsigned long positiveCount(const std::string& value, const std::string& option) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument(option + " requires a positive integer");
    const auto count = std::stoull(value);
    // CUDA launch calculations and existing implementations use signed int indices.
    if (count == 0 || count > static_cast<unsigned long>(std::numeric_limits<int>::max() - 1024))
        throw std::invalid_argument(option + " is outside the supported positive integer range");
    return static_cast<unsigned long>(count);
}

void argsReader(int argc, char **argv) {
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--help" || option == "-h") {
            if (argc != 2) throw std::invalid_argument("use --help on its own");
            printUsage();
            std::exit(EXIT_SUCCESS);
        }
        if (!seen.insert(option).second)
            throw std::invalid_argument("duplicate option: " + option);
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::invalid_argument("missing value for " + option);
            return argv[i];
        };
        if (option == "-n") NBodies = positiveCount(value(), option);
        else if (option == "-i") NIterations = positiveCount(value(), option);
        else if (option == "--im") ImplTag = value();
        else if (option == "--dt") {
            const std::string input = value();
            std::size_t consumed = 0;
            Dt = std::stof(input, &consumed);
            if (consumed != input.size() || !std::isfinite(Dt) || Dt <= 0)
                throw std::invalid_argument("--dt requires a finite positive number");
        }
        else if (option == "--nv") VisuEnable = false;
        else if (option == "--visu") VisuEnable = true;
        else if (option == "--gf") ShowGFlops = true;
        else if (option == "-v") Verbose = true;
        else if (option == "--scheme") BodiesScheme = value();
        else throw std::invalid_argument("unknown option: " + option + " (see --help)");
    }
    if (!seen.count("-n") || !seen.count("-i"))
        throw std::invalid_argument("both -n and -i are required (see --help)");
    if (seen.count("--nv") && seen.count("--visu"))
        throw std::invalid_argument("--nv and --visu are mutually exclusive");
#ifndef VISU
    if (seen.count("--visu"))
        throw std::invalid_argument("--visu requires a build with OpenGL visualization enabled");
#endif
    if (BodiesScheme != "galaxy" && BodiesScheme != "random")
        throw std::invalid_argument("--scheme must be galaxy or random in Phase 1");
    if (ImplTag != "cpu+naive" && ImplTag != "cpu+omp" && ImplTag != "gpu+tile+full")
        throw std::invalid_argument("unsupported Phase 1 backend: " + ImplTag);
#ifndef USE_CUDA
    if (ImplTag == "gpu+tile+full")
        throw std::invalid_argument("gpu+tile+full requires a CUDA build");
#endif
#ifndef _OPENMP
    if (ImplTag == "cpu+omp")
        throw std::invalid_argument("cpu+omp requires an OpenMP build");
#endif
}

std::string strDate(float timestamp) {
    unsigned int days = timestamp / (24 * 60 * 60);
    float rest = timestamp - (days * 24 * 60 * 60);
    unsigned int hours = rest / (60 * 60);
    rest = rest - (hours * 60 * 60);
    unsigned int minutes = rest / 60;
    rest = rest - (minutes * 60);
    std::stringstream res;
    res << std::setw(2) << days << "d " << std::setw(2) << hours << "h " << std::setw(2) << minutes << "m " << std::fixed << std::setprecision(2) << rest << "s";
    return res.str();
}

template <typename T>
SimulationNBodyInterface<T> *createImplem() {
    BodiesAllocator<T> allocator(NBodies, BodiesScheme);
    if (ImplTag == "cpu+naive") return new SimulationNBodyNaive<T>(allocator, Softening);
    if (ImplTag == "cpu+omp")   return new SimulationNBodyOpenMP<T>(allocator, Softening);
#ifdef USE_CUDA
    if (ImplTag == "gpu+tile+full") {
        CUDABodiesAllocator<T> cudaAllocator(NBodies, BodiesScheme);
        return new SimulationNBodyCUDATileFullDevice<T>(cudaAllocator, Softening, false);
    }
#endif
    throw std::invalid_argument("unsupported or unavailable Phase 1 backend: " + ImplTag);
}

template <typename T>
SpheresVisu *createVisu(SimulationNBodyInterface<T> *simu) {
#ifndef VISU
    return new SpheresVisuNo<T>();
#else
    if (!VisuEnable) return new SpheresVisuNo<T>();
    const T *px = simu->getBodies()->getDataSoA().qx.data();
    const T *py = simu->getBodies()->getDataSoA().qy.data();
    const T *pz = simu->getBodies()->getDataSoA().qz.data();
    const T *vx = simu->getBodies()->getDataSoA().vx.data();
    const T *vy = simu->getBodies()->getDataSoA().vy.data();
    const T *vz = simu->getBodies()->getDataSoA().vz.data();
    const T *r  = simu->getBodies()->getDataSoA().r.data();
    return new OGLSpheresVisuGS<T>("MUrB n-body", WinWidth, WinHeight, px, py, pz, vx, vy, vz, r, NBodies, VisuColor);
#endif
}

template <typename T>
void exportBinaryFrame(std::ofstream &outFile, SimulationNBodyInterface<T> *simu, unsigned long NBodies) {
    const dataSoA_t<T>& data = simu->getBodies()->getDataSoA();
    outFile.write(reinterpret_cast<const char*>(data.qx.data()), NBodies * sizeof(T));
    outFile.write(reinterpret_cast<const char*>(data.qy.data()), NBodies * sizeof(T));
    outFile.write(reinterpret_cast<const char*>(data.qz.data()), NBodies * sizeof(T));
}

#ifdef USE_CUDA
void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

void initializeCudaDevice() {
    int count = 0;
    checkCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count < 1) throw std::runtime_error("gpu+tile+full requires a visible CUDA GPU");
    // Phase 1 selects the first visible device; Slurm assigns one GPU to the task.
    // This must precede construction of any CUDA allocator or simulation object.
    checkCuda(cudaSetDevice(0), "cudaSetDevice(0)");
    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    char busId[32]{};
    checkCuda(cudaDeviceGetPCIBusId(busId, sizeof(busId), 0), "cudaDeviceGetPCIBusId");
    std::cout << "CUDA device: 0 (" << properties.name << "), PCI " << busId
              << ", visible devices: " << count << '\n';
}
#endif

int runSimulation(int argc, char **argv) {
    argsReader(argc, argv);
    int rank = 0;
#ifdef USE_CUDA
    if (ImplTag == "gpu+tile+full") initializeCudaDevice();
#endif

    SimulationNBodyInterface<float> *simu = createImplem<float>();
    NBodies = simu->getBodies()->getN();

    // --- PROTEZIONE PLAYER: Non aprire il file in scrittura se stiamo leggendo ---
    std::ofstream dumpFile;
    if (rank == 0 && ImplTag != "bin+player") {
        dumpFile.open("simulation_data.bin", std::ios::binary);
        if (dumpFile.is_open()) {
            dumpFile.write(reinterpret_cast<char*>(&NBodies), sizeof(unsigned long));
            dumpFile.write(reinterpret_cast<char*>(&NIterations), sizeof(unsigned long));
        }
    }

    SpheresVisu *visu = createVisu<float>(simu);
    simu->setDt(Dt);

    std::cout << "Simulation started..." << std::endl;
    Perf perfIte, perfTotal;
    float physicTime = 0.f;
    unsigned long iIte;

    for (iIte = 1; iIte <= NIterations && !visu->windowShouldClose(); iIte++) {
        visu->refreshDisplay();
        perfIte.start();
        simu->computeOneIteration();
        perfIte.stop();
        perfTotal += perfIte;

        // --- PROTEZIONE PLAYER: Non esportare se stiamo leggendo ---
        if (rank == 0 && ImplTag != "bin+player" && dumpFile.is_open() && iIte % 10 == 0) {
            exportBinaryFrame(dumpFile, simu, NBodies);
        }

        physicTime += simu->getDt();
        if (Verbose) std::cout << "Iteration n°" << iIte << " (" << perfTotal.getFPS(iIte) << " FPS)\r" << std::flush;
    }

    if (rank == 0 && dumpFile.is_open()) dumpFile.close();

    delete visu;
    delete simu;
    return EXIT_SUCCESS;
}
int main(int argc, char **argv) {
    try {
        return runSimulation(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "murb: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
