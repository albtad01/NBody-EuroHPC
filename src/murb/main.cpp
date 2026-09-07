#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
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
#include "utils/ArgumentsReader.hpp"
#include "utils/Perf.hpp"

#include "implem/SimulationNBodyNaive.hpp"
#include "implem/SimulationNBodyNop.hpp"
#include "implem/SimulationNBodyOptim.hpp"
#include "implem/SimulationNBodySIMD.hpp"
#include "implem/SimulationNBodyOpenMP.hpp" 
#include "implem/SimulationNBodyMultiNode.hpp"
#include "implem/SimulationNBodyBinaryPlayer.hpp"

#ifdef USE_CUDA
#include <mpi.h>
#include <cuda_runtime.h>
#include "implem/SimulationNBodyHetero.hpp"
#include "implem/SimulationNBodyCUDATile.hpp"
#include "implem/SimulationNBodyCUDATileFullDevice.hpp"
#include "implem/SimulationNBodyCUDATileFullDevice200k.hpp"
#include "implem/SimulationNBodyCUDAPropertyTracking.hpp"
#include "implem/SimulationNBodyCUDALeapfrog.hpp"
#include "implem/SimulationNBodyMultiNodeCUDA.hpp"
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

void argsReader(int argc, char **argv) {
    std::map<std::string, std::string> reqArgs, faculArgs, docArgs;
    Arguments_reader argsReader(argc, argv);
    reqArgs["n"] = "nBodies";
    reqArgs["i"] = "nIterations";
    faculArgs["-im"] = "ImplTag";
    faculArgs["-dt"] = "timeStep";
    faculArgs["-nv"] = "";
    faculArgs["-v"] = "";
    faculArgs["-gf"] = "";
    faculArgs["s"] = "scheme";

    if (argsReader.parse_arguments(reqArgs, faculArgs)) {
        NBodies = stoi(argsReader.get_argument("n"));
        NIterations = stoi(argsReader.get_argument("i"));
    } else { exit(-1); }

    if (argsReader.exist_argument("-v")) Verbose = true;
    if (argsReader.exist_argument("-im")) ImplTag = argsReader.get_argument("-im");
    if (argsReader.exist_argument("-nv")) VisuEnable = false;
    if (argsReader.exist_argument("-gf")) ShowGFlops = true;
    if (argsReader.exist_argument("s")) BodiesScheme = argsReader.get_argument("s");
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
    if (ImplTag == "mpi")       return new SimulationNBodyMultiNode<T>(allocator, Softening);
    if (ImplTag == "bin+player") return new SimulationNBodyBinaryPlayer<T>(allocator, Softening, "simulation_data.bin");
#ifdef USE_CUDA
    if (ImplTag == "gpu+multinode") {
        CUDABodiesAllocator<T> cudaAllocator(NBodies, BodiesScheme);
        return new SimulationNBodyMultiNodeCUDA<T>(cudaAllocator, Softening);
    }
#endif
    return nullptr;
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

int main(int argc, char **argv) {
    argsReader(argc, argv);
    int rank = 0;
#ifdef USE_CUDA
    int mpi_inited = 0;
    MPI_Initialized(&mpi_inited);
    if (!mpi_inited) MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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