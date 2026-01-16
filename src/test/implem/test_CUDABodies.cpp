#include <algorithm>
#include <catch.hpp>
#include <cmath>
#include <exception>
#include <numeric>
#include <random>
#include <string>
#include <iostream>

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

TEST_CASE("CUDABodies", "[cudabds]")
{
    // ===================================== Random configuration ============================================
    SECTION("fp32 - n=4000 - random") { test_cuda_bodies(4000, "random"); }
    SECTION("fp32 - n=4000 - galaxy") { test_cuda_bodies(4000, "galaxy"); }
}
