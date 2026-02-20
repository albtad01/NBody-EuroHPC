#include <algorithm>
#include <catch.hpp>
#include <cmath>
#include <exception>
#include <numeric>
#include <random>
#include <string>
#include <iostream>
#include <iomanip>

// Common CPU includes
#include "SimulationNBodyNaive.hpp"
#include "SimulationNBodyOpenMP.hpp"

// CUDA includes (only if enabled)
#ifdef USE_CUDA
    #include "SimulationNBodyCUDATileFullDevice.hpp"
#endif

template <typename T>
void compare_arrays(const T* a1, const T* a2, int n) {
    for ( int i = 0; i < n; i++ ) {
        CAPTURE(i);
        REQUIRE_THAT(a1[i], Catch::Matchers::WithinRel(a2[i]));
    }
}

void test_nbody_correctness(const size_t n, const float soft, const float dt, const size_t nIte, const std::string &scheme, const float eps)
{
    // Reference Implementation (CPU Naive)
    BodiesAllocator<float> naiveAllocator(n, scheme);
    SimulationNBodyNaive<float> simuRef(naiveAllocator, soft);
    simuRef.setDt(dt);

    // Target Implementation (GPU if CUDA is ON, otherwise CPU OpenMP)
#ifdef USE_CUDA
    CUDABodiesAllocator<float> targetAllocator(n, scheme);
    SimulationNBodyCUDATileFullDevice<float> simuTest(targetAllocator, soft);
#else
    BodiesAllocator<float> targetAllocator(n, scheme);
    SimulationNBodyOpenMP<float> simuTest(targetAllocator, soft);
#endif
    simuTest.setDt(dt);

    std::cout << std::setprecision(10);

    // Iteration loop
    for (size_t i = 0; i < nIte + 1; i++) {
        if (i > 0) { 
            simuRef.computeOneIteration(); 
            simuTest.computeOneIteration(); 
        }

        const float *xRef = simuRef.getBodies()->getDataSoA().qx.data();
        const float *yRef = simuRef.getBodies()->getDataSoA().qy.data();
        const float *zRef = simuRef.getBodies()->getDataSoA().qz.data();

        // On GPU, this call automatically triggers cudaMemcpy from Device to Host
        const float *xTest = simuTest.getBodies()->getDataSoA().qx.data();
        const float *yTest = simuTest.getBodies()->getDataSoA().qy.data();
        const float *zTest = simuTest.getBodies()->getDataSoA().qz.data();

        const float e = (i > 0) ? eps : 0.f;
        for (size_t b = 0; b < n; b++) {
            CAPTURE(b, i, std::log10(eps), xRef[b], xTest[b]);
            REQUIRE_THAT(xRef[b], Catch::Matchers::WithinRel(xTest[b], e));
            REQUIRE_THAT(yRef[b], Catch::Matchers::WithinRel(yTest[b], e));
            REQUIRE_THAT(zRef[b], Catch::Matchers::WithinRel(zTest[b], e));
        }
    }
}

TEST_CASE("n-body - Correctness", "[correctness]")
{
    // Random tests
    SECTION("fp32 - n=2048 - i=1 - random") { test_nbody_correctness(2048, 2e+08, 3600, 1, "random", 1e-3); }
    SECTION("fp32 - n=2049 - i=3 - random") { test_nbody_correctness(2049, 2e+08, 3600, 3, "random", 1e-3); }

    // Galaxy tests
    SECTION("fp32 - n=2048 - i=4 - galaxy") { test_nbody_correctness(2048, 2e+08, 3600, 4, "galaxy", 1e-1); }
    SECTION("fp32 - n=2049 - i=3 - galaxy") { test_nbody_correctness(2049, 2e+08, 3600, 3, "galaxy", 1e-1); }
}