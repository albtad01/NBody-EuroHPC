#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <fstream>
#include <memory>

#include "ogl/SpheresVisu.hpp"
#include "ogl/SpheresVisuNo.hpp"
#ifdef VISU
#include "ogl/OGLSpheresVisuGS.hpp"
#include "ogl/OGLSpheresVisuInst.hpp"
#endif

#include "core/Bodies.hpp"
#include "utils/Perf.hpp"
#include "BuildInfo.hpp"

#include "implem/SimulationNBodyNaive.hpp"
#include "implem/SimulationNBodyOpenMP.hpp"
#include "implem/SimulationNBodyOptim.hpp"
#include "implem/SimulationNBodySIMD.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "implem/SimulationNBodyCUDATile.hpp"
#include "implem/SimulationNBodyCUDATileFullDevice.hpp"
#include "implem/SimulationNBodyCUDATileFullDevice200k.hpp"
#endif

/* global variables */
unsigned long NBodies;               /*!< Number of bodies. */
unsigned long NIterations;           /*!< Number of iterations. */
std::string ImplTag = "cpu+naive";   /*!< Implementation id. */
bool Verbose = false;                /*!< Mode verbose. */
bool VisuEnable = false;              /*!< Enable visualization. */
bool VisuColor = true;               /*!< Enable visualization with colors. */
float Dt = 3600;                     /*!< Time step in seconds. */
float Softening = 2e+08;             /*!< Softening factor value. */
unsigned int WinWidth = 1024;        /*!< Window width for visualization. */
unsigned int WinHeight = 768;        /*!< Window height for visualization. */
std::string BodiesScheme = "galaxy"; /*!< Initial condition of the bodies. */
bool ShowGFlops = false;             /*!< Display the GFlop/s. */

bool isCudaBackend() {
    return ImplTag == "gpu+tile" || ImplTag == "gpu+tile+full" ||
           ImplTag == "gpu+tile+full200k";
}

void printVersion() {
    std::cout << "murb revision=" << MURB_REVISION << " dirty=" << MURB_BUILD_DIRTY
#ifdef USE_CUDA
              << " cuda=1"
#else
              << " cuda=0"
#endif
#ifdef _OPENMP
              << " openmp=1"
#else
              << " openmp=0"
#endif
#ifdef VISU
              << " visu=1"
#else
              << " visu=0"
#endif
              << '\n';
}

void printUsage() {
    std::cout << "Usage: murb -n BODIES -i ITERATIONS [--im BACKEND] [--nv] [--gf] [--dt SECONDS]\n"
              << "Exploratory backends: cpu+naive, cpu+optim, cpu+simd, cpu+omp, "
                 "gpu+tile, gpu+tile+full, gpu+tile+full200k (CUDA build only).\n"
              << "Default: headless, no recording. Options: --visu (local OpenGL), --scheme galaxy|random, -v, --help, --version.\n"
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
        if (option == "--version") {
            if (argc != 2) throw std::invalid_argument("use --version on its own");
            printVersion();
            std::exit(EXIT_SUCCESS);
        }
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
    if (ImplTag != "cpu+naive" && ImplTag != "cpu+optim" && ImplTag != "cpu+simd" &&
        ImplTag != "cpu+omp" && ImplTag != "gpu+tile" &&
        ImplTag != "gpu+tile+full" && ImplTag != "gpu+tile+full200k")
        throw std::invalid_argument("unsupported exploratory backend: " + ImplTag);
#ifndef USE_CUDA
    if (isCudaBackend())
        throw std::invalid_argument(ImplTag + " requires a CUDA build");
#endif
#ifndef _OPENMP
    if (ImplTag == "cpu+omp")
        throw std::invalid_argument("cpu+omp requires an OpenMP build");
#endif
}

template <typename T>
SimulationNBodyInterface<T> *createImplem() {
    BodiesAllocator<T> allocator(NBodies, BodiesScheme);
    if (ImplTag == "cpu+naive") return new SimulationNBodyNaive<T>(allocator, Softening);
    if (ImplTag == "cpu+optim") return new SimulationNBodyOptim<T>(allocator, Softening);
    if (ImplTag == "cpu+simd")  return new SimulationNBodySIMD<T>(allocator, Softening);
    if (ImplTag == "cpu+omp")   return new SimulationNBodyOpenMP<T>(allocator, Softening);
#ifdef USE_CUDA
    if (ImplTag == "gpu+tile") return new SimulationNBodyCUDATile<T>(allocator, Softening);
    if (ImplTag == "gpu+tile+full") {
        CUDABodiesAllocator<T> cudaAllocator(NBodies, BodiesScheme);
        return new SimulationNBodyCUDATileFullDevice<T>(cudaAllocator, Softening, false);
    }
    if (ImplTag == "gpu+tile+full200k") {
        CUDABodiesAllocator<T> cudaAllocator(NBodies, BodiesScheme);
        return new SimulationNBodyCUDATileFullDevice200k<T>(cudaAllocator, Softening, false);
    }
#endif
    throw std::invalid_argument("unsupported or unavailable exploratory backend: " + ImplTag);
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
// Preserved for Phase 1b; the Phase 1 runner never opens a trajectory file.
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
    printVersion();
#ifdef USE_CUDA
    if (isCudaBackend()) initializeCudaDevice();
#endif

    std::unique_ptr<SimulationNBodyInterface<float>> simu(createImplem<float>());
    NBodies = simu->getBodies()->getN();
    std::unique_ptr<SpheresVisu> visu(createVisu<float>(simu.get()));
    simu->setDt(Dt);

    std::cout << "backend=" << ImplTag << " bodies=" << NBodies
              << " iterations=" << NIterations << " dt=" << std::setprecision(9) << Dt
              << " softening=" << Softening << " scheme=" << BodiesScheme
              << " precision=fp32 visualization=" << (VisuEnable ? "on" : "off")
              << " recording=off\n";

    Perf perfIte, perfTotal, wall;
    unsigned long completed = 0;
#ifdef USE_CUDA
    if (isCudaBackend())
        checkCuda(cudaDeviceSynchronize(), "CUDA completion before timing");
#endif
    wall.start();
    while (completed < NIterations && !visu->windowShouldClose()) {
        perfIte.start();
        simu->computeOneIteration();
#ifdef USE_CUDA
        if (isCudaBackend()) {
            checkCuda(cudaGetLastError(), "CUDA iteration launch");
            checkCuda(cudaDeviceSynchronize(), "CUDA iteration completion");
        }
#endif
        perfIte.stop();
        perfTotal += perfIte;
        ++completed;

        if (VisuEnable) {
            // Refresh cached GPU state outside compute timing, before rendering.
            simu->getBodies()->getDataSoA();
            visu->refreshDisplay();
        }
        if (Verbose && (completed == 1 || completed % 10 == 0 || completed == NIterations))
            std::cout << "completed=" << completed << '/' << NIterations << '\n';
    }
    wall.stop();
    const double interactions = static_cast<double>(NBodies) * NBodies * completed;
    const double computeMs = perfTotal.getElapsedTime();
    const double rate = computeMs > 0 ? interactions * 1000.0 / computeMs : 0.0;
    std::cout << std::setprecision(9)
              << "completed_iterations=" << completed
              << " compute_ms=" << computeMs
              << " loop_wall_ms=" << wall.getElapsedTime()
              << " interactions_per_second=" << rate;
    if (ShowGFlops)
        std::cout << " estimated_GFLOP_per_second=" << rate * 20.0 / 1e9;
    std::cout << '\n';
    if (computeMs <= 0) std::cout << "Timing below clock resolution; rates reported as zero.\n";
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
