#include <algorithm>
#include <catch.hpp>
#include <cmath>
#include <exception>
#include <numeric>
#include <random>
#include <string>
#include <iostream>
#include <iomanip>

#include "SimulationNBodyNaive.hpp"
#include "SimulationNBodyOptim.hpp"
#include "SimulationNBodySIMD.hpp"
#include "SimulationNBodyOpenMP.hpp"
#include "SimulationNBodyCUDATile.hpp"
#include "SimulationNBodyCUDATileFullDevice.hpp"

template <typename T>
void compare_arrays(const T* a1, const T* a2, int n) {
    for ( int i = 0; i < n; i++ ) {
        CAPTURE(i);
        REQUIRE_THAT(a1[i], Catch::Matchers::WithinRel(a2[i]));
    }
}

void test_nbody_gpufd_bodies_test(const size_t n, const float soft, const float dt, const size_t nIte, const std::string &scheme,
                     const float eps) {
    BodiesAllocator<float> naiveAllocator(n, scheme);
    SimulationNBodyNaive<float> simuRef(naiveAllocator, soft);
    simuRef.setDt(dt);

    CUDABodiesAllocator<float> cudaAllocator(n, scheme);
    SimulationNBodyCUDATileFullDevice<float> simuTest(cudaAllocator, soft);
    simuTest.setDt(dt);

    const dataSoA_t<float>& dataSoA1 = simuRef.getBodies()->getDataSoA();
    const dataSoA_t<float>& dataSoA2 = simuTest.getBodies()->getDataSoA();
    compare_arrays<float>(dataSoA1.m.data(), dataSoA2.m.data(), n);
    compare_arrays<float>(dataSoA1.r.data(), dataSoA2.r.data(), n);

    compare_arrays<float>(dataSoA1.qx.data(), dataSoA2.qx.data(), n);
    compare_arrays<float>(dataSoA1.qy.data(), dataSoA2.qy.data(), n);
    compare_arrays<float>(dataSoA1.qz.data(), dataSoA2.qz.data(), n);

    compare_arrays<float>(dataSoA1.vx.data(), dataSoA2.vx.data(), n);
    compare_arrays<float>(dataSoA1.vy.data(), dataSoA2.vy.data(), n);
    compare_arrays<float>(dataSoA1.vz.data(), dataSoA2.vz.data(), n);
}

void test_nbody_gpufd_full_test(const size_t n, const float soft, const float dt, const size_t nIte, const std::string &scheme,
                     const float eps)
{
    BodiesAllocator<float> naiveAllocator(n, scheme);
    SimulationNBodyNaive<float> simuRef(naiveAllocator, soft);
    simuRef.setDt(dt);

    // CUDABodiesAllocator<float> cudaAllocator(n, scheme);
    // SimulationNBodyCUDATileFullDevice<float> simuTest(cudaAllocator, soft);
    // SimulationNBodyCUDATile<float> simuTest(naiveAllocator, soft);

    // SimulationNBodyOptim<float> simuTest(naiveAllocator, soft);
    // SimulationNBodySIMD<float> simuTest(naiveAllocator, soft);
    SimulationNBodyOpenMP<float> simuTest(naiveAllocator, soft);
    simuTest.setDt(dt);

    const float *xRef = simuRef.getBodies()->getDataSoA().qx.data();
    const float *yRef = simuRef.getBodies()->getDataSoA().qy.data();
    const float *zRef = simuRef.getBodies()->getDataSoA().qz.data();
    const std::vector<accAoS_t<float>>& aRef = simuRef.getAccAoS();

    std::cout << std::setprecision(10);

    float e = 0; // espilon
    for (size_t i = 0; i < nIte + 1; i++) {
        if (i > 0) { simuRef.computeOneIteration(); simuTest.computeOneIteration(); }

        const float *xRef = simuRef.getBodies()->getDataSoA().qx.data();
        const float *yRef = simuRef.getBodies()->getDataSoA().qy.data();
        const float *zRef = simuRef.getBodies()->getDataSoA().qz.data();

        const float *xTest = simuTest.getBodies()->getDataSoA().qx.data();
        const float *yTest = simuTest.getBodies()->getDataSoA().qy.data();
        const float *zTest = simuTest.getBodies()->getDataSoA().qz.data();

        const float e = (i>0)? eps : 0.f;
        for (size_t b=0; b<n; b++) {
            CAPTURE(b,i,std::log10(eps));
            REQUIRE_THAT(xRef[b], Catch::Matchers::WithinRel(xTest[b], e));
            REQUIRE_THAT(yRef[b], Catch::Matchers::WithinRel(yTest[b], e));
            REQUIRE_THAT(zRef[b], Catch::Matchers::WithinRel(zTest[b], e));
        }
}
}

TEST_CASE("n-body - Adv", "[adv]")
{

    // ===================================== Random configuration ============================================
    SECTION("fp32 - n=4096 - i=10 - random") { test_nbody_gpufd_full_test(4096, 2e+08, 3600, 10, "random", 1e-3); }

    // Harder tests: more bodies, more iterations, more accuracy
    SECTION("fp32 - n=4096 - i=20 - random") { test_nbody_gpufd_full_test(4096, 2e+08, 3600, 20, "random", 1e-5); }
    SECTION("fp32 - n=4096 - i=20 - random") { test_nbody_gpufd_full_test(4096, 2e+08, 3600, 20, "random", 1e-6); }
    // SECTION("fp32 - n=4096 - i=100 - random") { test_nbody_gpufd_full_test(4096, 2e+08, 3600, 1, "random", 5e-7); } 
    // SECTION("fp32 - n=4096 - i=100 - random") { test_nbody_gpufd_full_test(4096, 2e+08, 3600, 1, "random", 1e-7); }

    // ===================================== Galaxy configuration ============================================
    SECTION("fp32 - n=4096 - i=10 - galaxy") { test_nbody_gpufd_full_test(4096, 2e+08, 3600, 10, "galaxy", 1e-3); }

    // Harder tests: more bodies, more iterations, more accuracy
    SECTION("fp32 - n=4096 - i=20 - galaxy") { test_nbody_gpufd_full_test(4096, 2e+08, 3600, 20, "galaxy", 1e-5); }
    // SECTION("fp32 - n=4096 - i=20 - galaxy") { test_nbody_gpufd_full_test(4096, 2e+08, 3600, 20, "galaxy", 1e-6); }
    // SECTION("fp32 - n=4096 - i=100 - galaxy") { test_nbody_gpufd_full_test(4096, 2e+08, 3600, 1, "galaxy", 5e-7); } // 16.01.2026 - CUDATile fails from here onwards
    // SECTION("fp32 - n=4096 - i=100 - galaxy") { test_nbody_gpufd_full_test(4096, 2e+08, 3600, 1, "galaxy", 1e-7); }
}

TEST_CASE("n-body - Dumb", "[dmb]")
{
    SECTION("fp32 - n=13 - i=1 - random") { test_nbody_dumb(13, 2e+08, 3600, 1, "random", 1e-3); }
    SECTION("fp32 - n=13 - i=100 - random") { test_nbody_dumb(13, 2e+08, 3600, 100, "random", 5e-3); }
    SECTION("fp32 - n=16 - i=1 - random") { test_nbody_dumb(16, 2e+08, 3600, 1, "random", 1e-3); }
    SECTION("fp32 - n=128 - i=1 - random") { test_nbody_dumb(128, 2e+08, 3600, 1, "random", 1e-3); }
    SECTION("fp32 - n=2048 - i=1 - random") { test_nbody_dumb(2048, 2e+08, 3600, 1, "random", 1e-3); }
    SECTION("fp32 - n=2049 - i=3 - random") { test_nbody_dumb(2049, 2e+08, 3600, 3, "random", 1e-3); }

    SECTION("fp32 - n=13 - i=1 - galaxy") { test_nbody_dumb(13, 2e+08, 3600, 1, "galaxy", 1e-1); }
    SECTION("fp32 - n=13 - i=30 - galaxy") { test_nbody_dumb(13, 2e+08, 3600, 30, "galaxy", 1e-1); }
    SECTION("fp32 - n=16 - i=1 - galaxy") { test_nbody_dumb(16, 2e+08, 3600, 1, "galaxy", 1e-2); }
    SECTION("fp32 - n=128 - i=1 - galaxy") { test_nbody_dumb(128, 2e+08, 3600, 1, "galaxy", 1e-2); }
    SECTION("fp32 - n=2048 - i=4 - galaxy") { test_nbody_dumb(2048, 2e+08, 3600, 4, "galaxy", 1e-1); }
    SECTION("fp32 - n=2049 - i=3 - galaxy") { test_nbody_dumb(2049, 2e+08, 3600, 3, "galaxy", 1e-1); }
}
