
#include <algorithm>
#include <catch.hpp>
#include <cmath>
#include <exception>
#include <numeric>
#include <random>
#include <string>
#include <iostream>

#ifdef USE_CUDA
#include "core/Bodies.hpp"
#include "core/CUDABodies.hpp"

template <typename T>
void compare_arrays(const T* a1, const T* a2, int n) {
    for ( int i = 0; i < n; i++ ) {
        CAPTURE(i);
        REQUIRE_THAT(a1[i], Catch::Matchers::WithinRel(a2[i]));
    }
}

void test_cuda_bodies(const size_t n, const std::string &scheme)
{
    Bodies<float> bodies(n, scheme);
    CUDABodies<float> cudaBodies(n, scheme);

    const dataSoA_t<float>& dataSoA1 = bodies.getDataSoA();
    const dataSoA_t<float>& dataSoA2 = cudaBodies.getDataSoA();
    compare_arrays<float>(dataSoA1.m.data(), dataSoA2.m.data(), n);
    compare_arrays<float>(dataSoA1.r.data(), dataSoA2.r.data(), n);

    compare_arrays<float>(dataSoA1.qx.data(), dataSoA2.qx.data(), n);
    compare_arrays<float>(dataSoA1.qy.data(), dataSoA2.qy.data(), n);
    compare_arrays<float>(dataSoA1.qz.data(), dataSoA2.qz.data(), n);

    compare_arrays<float>(dataSoA1.vx.data(), dataSoA2.vx.data(), n);
    compare_arrays<float>(dataSoA1.vy.data(), dataSoA2.vy.data(), n);
    compare_arrays<float>(dataSoA1.vz.data(), dataSoA2.vz.data(), n);
}

void test_cuda_bodies_update(const size_t n, const std::string &scheme)
{
    Bodies<float> bodies(n, scheme);
    CUDABodies<float> cudaBodies(n, scheme);

    accSoA_t<float> acc;
    acc.ax.resize(n);
    acc.ay.resize(n);
    acc.az.resize(n);
    
    for(unsigned long i = 0; i < n; i++) {
        acc.ax[i] = i + 1;
        acc.ay[i] = 3.0f;
        acc.az[i] = n - i;
    }
    
    float dt = 0.01f;
    for ( int i = 0; i < 4; i++ ) {
        bodies.updatePositionsAndVelocities(acc, dt);
        cudaBodies.updatePositionsAndVelocities(acc, dt);
        
        const dataSoA_t<float>& dataSoA1 = bodies.getDataSoA();
        const dataSoA_t<float>& dataSoA2 = cudaBodies.getDataSoA();
        
        compare_arrays<float>(dataSoA1.qx.data(), dataSoA2.qx.data(), n);
        compare_arrays<float>(dataSoA1.qy.data(), dataSoA2.qy.data(), n);
        compare_arrays<float>(dataSoA1.qz.data(), dataSoA2.qz.data(), n);

        compare_arrays<float>(dataSoA1.vx.data(), dataSoA2.vx.data(), n);
        compare_arrays<float>(dataSoA1.vy.data(), dataSoA2.vy.data(), n);
        compare_arrays<float>(dataSoA1.vz.data(), dataSoA2.vz.data(), n);
    }

}

TEST_CASE("CUDABodies", "[cudabds]")
{
    SECTION("fp32 - n=4000 - random") { test_cuda_bodies(4000, "random"); }
    SECTION("fp32 - n=4000 - galaxy") { test_cuda_bodies(4000, "galaxy"); }
    SECTION("fp32 - n=4000 - random - update") { test_cuda_bodies_update(4000, "random"); }
    SECTION("fp32 - n=4000 - galaxy - update") { test_cuda_bodies_update(4000, "galaxy"); }
}
#endif