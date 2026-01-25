#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

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
#include "implem/SimulationNBodyHetero.hpp"
// #include "implem/SimulationNBodyMultiNode.hpp"
#include "implem/SimulationNBodyCUDATile.hpp"
#include "implem/SimulationNBodyCUDATileFullDevice.hpp"
#include "implem/SimulationNBodyCUDATileFullDevice200k.hpp"
#include "implem/SimulationNBodyCUDAPropertyTracking.hpp"
#include "implem/SimulationNBodyCUDALeapfrog.hpp"

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

/*!
 * \fn     void argsReader(int argc, char** argv)
 * \brief  Read arguments from command line and set global variables.
 *
 * \param  argc : Number of arguments.
 * \param  argv : Array of arguments.
 */
void argsReader(int argc, char **argv)
{
    std::map<std::string, std::string> reqArgs, faculArgs, docArgs;
    Arguments_reader argsReader(argc, argv);

    reqArgs["n"] = "nBodies";
    docArgs["n"] = "the number of generated bodies.";
    reqArgs["i"] = "nIterations";
    docArgs["i"] = "the number of iterations to compute.";

    faculArgs["v"] = "";
    docArgs["v"] = "enable verbose mode.";
    faculArgs["h"] = "";
    docArgs["h"] = "display this help.";
    faculArgs["-help"] = "";
    docArgs["-help"] = "display this help.";
    faculArgs["-dt"] = "timeStep";
    docArgs["-dt"] = "select a fixed time step in second (default is " + std::to_string(Dt) + " sec).";
    faculArgs["-ngs"] = "";
    docArgs["-ngs"] = "disable geometry shader for visu (slower but it should work with old GPUs).";
    faculArgs["-ww"] = "winWidth";
    docArgs["-ww"] = "the width of the window in pixel (default is " + std::to_string(WinWidth) + ").";
    faculArgs["-wh"] = "winHeight";
    docArgs["-wh"] = "the height of the window in pixel (default is " + std::to_string(WinHeight) + ").";
    faculArgs["-nv"] = "";
    docArgs["-nv"] = "no visualization (disable visu).";
    faculArgs["-nvc"] = "";
    docArgs["-nvc"] = "visualization without colors.";
    faculArgs["-im"] = "ImplTag";
    docArgs["-im"] = "code implementation tag:\n"
                     "\t\t\t - \"cpu+naive\"\n"
                     "\t\t\t - \"cpu+optim\"\n"
                     "\t\t\t - \"cpu+simd\"\n"
                     "\t\t\t - \"cpu+omp\n"
                     "\t\t\t - \"hetero\"\n"
                    //  "\t\t\t - \"mpi\"\n"
                     "\t\t\t - \"gpu+tile\"\n"
                     "\t\t\t - \"gpu+tile+full\n"
                     "\t\t\t - \"gpu+tracking\n"
                     "\t\t\t - \"gpu+leapfrog\n"
                     "\t\t\t ----";
    faculArgs["-soft"] = "softeningFactor";
    docArgs["-soft"] = "softening factor.";
#ifdef USE_OCL
    faculArgs["-wg"] = "workGroup";
    docArgs["-wg"] = "the size of the OpenCL local workgroup (default is " + std::to_string(LocalWGSize) + ").";
#endif
    faculArgs["s"] = "bodies scheme";
    docArgs["s"] = "bodies scheme (initial conditions can be \"galaxy\" or \"random\").";
    faculArgs["-gf"] = "";
    docArgs["-gf"] = "display the number of GFlop/s.";

    if (argsReader.parse_arguments(reqArgs, faculArgs)) {
        NBodies = stoi(argsReader.get_argument("n"));
        NIterations = stoi(argsReader.get_argument("i"));
    }
    else {
        if (argsReader.parse_doc_args(docArgs))
            argsReader.print_usage();
        else
            std::cout << "A problem was encountered when parsing arguments documentation... exiting." << std::endl;
        exit(-1);
    }

    if (argsReader.exist_argument("h") || argsReader.exist_argument("-help")) {
        if (argsReader.parse_doc_args(docArgs))
            argsReader.print_usage();
        else
            std::cout << "A problem was encountered when parsing arguments documentation... exiting." << std::endl;
        exit(-1);
    }

    if (argsReader.exist_argument("v"))
        Verbose = true;
    if (argsReader.exist_argument("-dt"))
        Dt = stof(argsReader.get_argument("-dt"));
    if (argsReader.exist_argument("-ngs"))
        GSEnable = false;
    if (argsReader.exist_argument("-ww"))
        WinWidth = stoi(argsReader.get_argument("-ww"));
    if (argsReader.exist_argument("-wh"))
        WinHeight = stoi(argsReader.get_argument("-wh"));
    if (argsReader.exist_argument("-nv"))
        VisuEnable = false;
    if (argsReader.exist_argument("-nvc"))
        VisuColor = false;
    if (argsReader.exist_argument("-im"))
        ImplTag = argsReader.get_argument("-im");
    if (argsReader.exist_argument("-soft")) {
        Softening = stof(argsReader.get_argument("-soft"));
        if (Softening == 0.f) {
            std::cout << "Softening factor can't be equal to 0... exiting." << std::endl;
            exit(-1);
        }
    }
#ifdef USE_OCL
    if (argsReader.exist_argument("-wg"))
        LocalWGSize = stoi(argsReader.get_argument("-wg"));
#endif
    if (argsReader.exist_argument("s"))
        BodiesScheme = argsReader.get_argument("s");
    if (argsReader.exist_argument("-gf"))
        ShowGFlops = true;
}

/*!
 * \fn     string strDate(float timestamp)
 * \brief  Convert a timestamp into a string "..d ..h ..m ..s".
 *
 * \param  Timestamp : The timestamp to convert
 *
 * \return Date as a string.
 */
std::string strDate(float timestamp)
{
    unsigned int days;
    unsigned int hours;
    unsigned int minutes;
    float rest;

    days = timestamp / (24 * 60 * 60);
    rest = timestamp - (days * 24 * 60 * 60);

    hours = rest / (60 * 60);
    rest = rest - (hours * 60 * 60);

    minutes = rest / 60;
    rest = rest - (minutes * 60);

    std::stringstream res;
    res << std::setprecision(0) << std::fixed << std::setw(4) << days << "d " << std::setprecision(0) << std::fixed
        << std::setw(4) << hours << "h " << std::setprecision(0) << std::fixed << std::setw(4) << minutes << "m "
        << std::setprecision(3) << std::fixed << std::setw(5) << rest << "s";

    return res.str();
}

/*!
 * \fn     SimulationNBodyInterface *createImplem()
 * \brief  Select and allocate an n-body simulation object.
 *
 * \return A fresh allocated simulation.
 */
template <typename T>
SimulationNBodyInterface<T> *createImplem()
{
    SimulationNBodyInterface<T> *simu = nullptr;
    
    BodiesAllocator<T> allocator(NBodies, BodiesScheme);

    if (ImplTag == "cpu+naive") {
        simu = new SimulationNBodyNaive<T>(allocator, Softening);
    }
    else if (ImplTag == "cpu+nop") {
        simu = new SimulationNBodyNop<T>(allocator, Softening);
    }
    else if (ImplTag == "cpu+optim") {
        simu = new SimulationNBodyOptim<T>(allocator, Softening);
    }
    else if (ImplTag == "cpu+simd") {
        simu = new SimulationNBodySIMD<T>(allocator, Softening);
    }
    else if (ImplTag == "cpu+omp") {
        simu = new SimulationNBodyOpenMP<T>(allocator, Softening);
    }
    else if (ImplTag == "hetero") {
        simu = new SimulationNBodyHetero<T>(allocator, Softening);
    }
    // else if (ImplTag == "mpi") {
    //     simu = new SimulationNBodyMultiNode<T>(allocator, Softening);
    // }
    else if (ImplTag == "gpu+tile") {
        simu = new SimulationNBodyCUDATile<T>(allocator, Softening);
    }
    else if (ImplTag == "gpu+tile+full") {
        CUDABodiesAllocator<T> cudaAllocator(NBodies, BodiesScheme);
        simu = new SimulationNBodyCUDATileFullDevice<T>(cudaAllocator, Softening);
    }
    else if (ImplTag == "gpu+tile+full200k") {
        CUDABodiesAllocator<T> cudaAllocator(NBodies, BodiesScheme);
        simu = new SimulationNBodyCUDATileFullDevice200k<T>(cudaAllocator, Softening);
    }
    else if (ImplTag == "gpu+tracking") {
        CUDABodiesAllocator<T> cudaAllocator(NBodies, BodiesScheme);
        std::shared_ptr<GPUSimulationHistory<double>> history 
            = std::make_shared<GPUSimulationHistory<double>>(NIterations);
        simu = new SimulationNBodyCUDAPropertyTracking<T,double>(cudaAllocator, 
                                                          history,   
                                                          Softening);
    }
    else if (ImplTag == "gpu+leapfrog") {
        CUDABodiesAllocator<T> cudaAllocator(NBodies, BodiesScheme);
        std::shared_ptr<GPUSimulationHistory<double>> history 
            = std::make_shared<GPUSimulationHistory<double>>(NIterations);
        simu = new SimulationNBodyCUDALeapfrog<T,double>(cudaAllocator, 
                                                          history,   
                                                          Softening,
                                                          NIterations,
                                                          true);
    }

    else {
        std::cout << "Implementation '" << ImplTag << "' does not exist... Exiting." << std::endl;
        exit(-1);
    }
    return simu;
}

template <typename T>
SpheresVisu *createVisu(SimulationNBodyInterface<T> *simu)
{
    SpheresVisu *visu;

#ifdef VISU
    if (VisuEnable) {
        const T *positionsX = simu->getBodies()->getDataSoA().qx.data();
        const T *positionsY = simu->getBodies()->getDataSoA().qy.data();
        const T *positionsZ = simu->getBodies()->getDataSoA().qz.data();

        const T *velocitiesX = simu->getBodies()->getDataSoA().vx.data();
        const T *velocitiesY = simu->getBodies()->getDataSoA().vy.data();
        const T *velocitiesZ = simu->getBodies()->getDataSoA().vz.data();

        const T *radiuses = simu->getBodies()->getDataSoA().r.data();

        if (GSEnable) // geometry shader = better performances on dedicated GPUs
            visu = new OGLSpheresVisuGS<T>("MUrB n-body (geometry shader)", WinWidth, WinHeight, positionsX,
                                               positionsY, positionsZ, velocitiesX, velocitiesY, velocitiesZ, radiuses,
                                               NBodies, VisuColor);
        else
            visu = new OGLSpheresVisuInst<T>("MUrB n-body (instancing)", WinWidth, WinHeight, positionsX,
                                                 positionsY, positionsZ, velocitiesX, velocitiesY, velocitiesZ,
                                                 radiuses, NBodies, VisuColor);
        std::cout << std::endl;
    }
    else
        visu = new SpheresVisuNo<T>();
#else
    VisuEnable = false;
    visu = new SpheresVisuNo<T>();
#endif

    return visu;
}

int main(int argc, char **argv)
{
    // read arguments from the command line
    // usage: ./nbody -n nBodies  -i nIterations [-v] [-w] ...
    argsReader(argc, argv);

    // create the n-body simulation
    SimulationNBodyInterface<float> *simu = createImplem<float>();
    NBodies = simu->getBodies()->getN();

    // get MB used for this simulation
    float Mbytes = simu->getAllocatedBytes() / 1024.f / 1024.f;

    // display simulation configuration
    std::cout << "n-body simulation configuration:" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "  -> bodies scheme     (-s    ): " << BodiesScheme << std::endl;
    std::cout << "  -> implementation    (--im  ): " << ImplTag << std::endl;
    std::cout << "  -> nb. of bodies     (-n    ): " << NBodies << std::endl;
    std::cout << "  -> nb. of iterations (-i    ): " << NIterations << std::endl;
    std::cout << "  -> verbose mode      (-v    ): " << ((Verbose) ? "enable" : "disable") << std::endl;
    std::cout << "  -> precision                 : " << "fp32" << std::endl;
    std::cout << "  -> mem. allocated            : " << Mbytes << " MB" << std::endl;
    std::cout << "  -> geometry shader   (--ngs ): " << ((GSEnable) ? "enable" : "disable") << std::endl;
    std::cout << "  -> time step         (--dt  ): " << std::to_string(Dt) + " sec" << std::endl;
    std::cout << "  -> softening factor  (--soft): " << Softening << std::endl;

    // initialize visualization of bodies (with spheres in space)
    SpheresVisu *visu = createVisu<float>(simu);

    // time step selection
    simu->setDt(Dt);

    std::cout << "Simulation started..." << std::endl;

    // loop over the iterations
    Perf perfIte, perfTotal;
    float physicTime = 0.f;
    unsigned long iIte;
    for (iIte = 1; iIte <= NIterations && !visu->windowShouldClose(); iIte++) {
        // refresh the display in OpenGL window
        visu->refreshDisplay();

        // simulation computations
        perfIte.start();
        simu->computeOneIteration();
        perfIte.stop();
        perfTotal += perfIte;

        // compute the elapsed physic time
        physicTime += simu->getDt();

        // display the status of this iteration
        if (Verbose) {
            std::stringstream gflops;
            if (ShowGFlops)
                gflops << ", " << std::setprecision(1) << std::fixed << std::setw(6)
                       << perfTotal.getGflops(simu->getFlopsPerIte() * iIte) << " Gflop/s";
            std::cout << "Iteration n°" << std::setw(4) << iIte << " (" << std::setprecision(1) << std::fixed
                      << std::setw(6) << perfTotal.getFPS(iIte) << " FPS" << gflops.str()
                      << "), physic time: " << strDate(physicTime) << "\r";
            if (iIte % 5 == 0)
                std::cout << std::flush;
        }
    }
    if (Verbose)
        std::cout << std::endl;

    std::cout << "Simulation ended." << std::endl << std::endl;

    std::stringstream gflops;
    if (ShowGFlops)
        gflops << ", " << std::setprecision(1) << std::fixed << std::setw(6)
               << perfTotal.getGflops(simu->getFlopsPerIte() * (iIte - 1)) << " Gflop/s";
    std::cout << "Entire simulation took " << perfTotal.getElapsedTime() << " ms "
              << "(" << perfTotal.getFPS(iIte - 1) << " FPS" << gflops.str() << ")" << std::endl;

    // ((SimulationNBodyCUDAPropertyTracking<double,double>*)simu)->getHistory()->copyFromDevice();
    // ((SimulationNBodyCUDAPropertyTracking<double,double>*)simu)->getHistory()->saveMetricsToCSV("metrics.csv");
    // free resources
    delete visu;
    delete simu;

    return EXIT_SUCCESS;
}
