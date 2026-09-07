#include <algorithm>
#include <catch.hpp>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "SimulationNBodyNaive.hpp"
#include "SimulationNBodyOpenMP.hpp"
#include "SimulationNBodyOptim.hpp"
#include "SimulationNBodySIMD.hpp"

#ifdef USE_CUDA
#include "SimulationNBodyCUDATile.hpp"
#include "SimulationNBodyCUDATileFullDevice.hpp"
#include "SimulationNBodyCUDATileFullDevice200k.hpp"
#endif

enum class Backend {
    CpuOptim,
    CpuSimd,
    CpuOmp,
#ifdef USE_CUDA
    GpuTile,
    GpuTileFull,
    GpuTileFull200k,
#endif
};

const char* backendName(Backend backend) {
    switch (backend) {
        case Backend::CpuOptim: return "cpu+optim";
        case Backend::CpuSimd: return "cpu+simd";
        case Backend::CpuOmp: return "cpu+omp";
#ifdef USE_CUDA
        case Backend::GpuTile: return "gpu+tile";
        case Backend::GpuTileFull: return "gpu+tile+full";
        case Backend::GpuTileFull200k: return "gpu+tile+full200k";
#endif
    }
    return "unknown";
}

std::unique_ptr<SimulationNBodyInterface<float>> makeTarget(
    Backend backend, const size_t n, const std::string& scheme, const float soft)
{
    BodiesAllocator<float> hostAllocator(n, scheme);
    switch (backend) {
        case Backend::CpuOptim:
            return std::make_unique<SimulationNBodyOptim<float>>(hostAllocator, soft);
        case Backend::CpuSimd:
            return std::make_unique<SimulationNBodySIMD<float>>(hostAllocator, soft);
        case Backend::CpuOmp:
            return std::make_unique<SimulationNBodyOpenMP<float>>(hostAllocator, soft);
#ifdef USE_CUDA
        case Backend::GpuTile:
            return std::make_unique<SimulationNBodyCUDATile<float>>(hostAllocator, soft);
        case Backend::GpuTileFull: {
            CUDABodiesAllocator<float> cudaAllocator(n, scheme);
            return std::make_unique<SimulationNBodyCUDATileFullDevice<float>>(cudaAllocator, soft, false);
        }
        case Backend::GpuTileFull200k: {
            CUDABodiesAllocator<float> cudaAllocator(n, scheme);
            return std::make_unique<SimulationNBodyCUDATileFullDevice200k<float>>(cudaAllocator, soft, false);
        }
#endif
    }
    throw std::invalid_argument("unknown test backend");
}

void compareValue(const float reference, const float candidate, const float tolerance) {
    const float scale = std::max(1.0f, std::max(std::abs(reference), std::abs(candidate)));
    REQUIRE(std::abs(reference - candidate) <= tolerance * scale);
}

void testNBodyCorrectness(Backend backend, const size_t n, const float soft, const float dt,
                          const size_t iterations, const std::string& scheme, const float tolerance)
{
    BodiesAllocator<float> naiveAllocator(n, scheme);
    SimulationNBodyNaive<float> referenceSimulation(naiveAllocator, soft);
    referenceSimulation.setDt(dt);

    auto candidateSimulation = makeTarget(backend, n, scheme, soft);
    candidateSimulation->setDt(dt);

    for (size_t iteration = 0; iteration <= iterations; ++iteration) {
        if (iteration > 0) {
            referenceSimulation.computeOneIteration();
            candidateSimulation->computeOneIteration();
        }

        const auto& reference = referenceSimulation.getBodies()->getDataSoA();
        const auto& candidate = candidateSimulation->getBodies()->getDataSoA();
        const float allowed = iteration > 0 ? tolerance : 0.0f;

        for (size_t body = 0; body < n; ++body) {
            CAPTURE(backendName(backend), scheme, n, body, iteration, tolerance);
            compareValue(reference.qx[body], candidate.qx[body], allowed);
            compareValue(reference.qy[body], candidate.qy[body], allowed);
            compareValue(reference.qz[body], candidate.qz[body], allowed);
            compareValue(reference.vx[body], candidate.vx[body], allowed);
            compareValue(reference.vy[body], candidate.vy[body], allowed);
            compareValue(reference.vz[body], candidate.vz[body], allowed);
        }
    }
}

void testBackend(Backend backend) {
    testNBodyCorrectness(backend, 2048, 2e+08f, 3600, 1, "random", 1e-3f);
    testNBodyCorrectness(backend, 2049, 2e+08f, 3600, 3, "random", 1e-3f);
    testNBodyCorrectness(backend, 2048, 2e+08f, 3600, 4, "galaxy", 1e-1f);
    testNBodyCorrectness(backend, 2049, 2e+08f, 3600, 3, "galaxy", 1e-1f);
}

TEST_CASE("cpu+optim matches cpu+naive", "[correctness][cpu-optim]") {
    testBackend(Backend::CpuOptim);
}

TEST_CASE("cpu+simd matches cpu+naive", "[correctness][cpu-simd]") {
    testBackend(Backend::CpuSimd);
}

TEST_CASE("cpu+omp matches cpu+naive", "[correctness][cpu-omp]") {
    testBackend(Backend::CpuOmp);
}

#ifdef USE_CUDA
TEST_CASE("gpu+tile matches cpu+naive", "[correctness][gpu-tile]") {
    testBackend(Backend::GpuTile);
}

TEST_CASE("gpu+tile+full matches cpu+naive", "[correctness][gpu-tile-full]") {
    testBackend(Backend::GpuTileFull);
}

TEST_CASE("gpu+tile+full200k matches cpu+naive", "[correctness][gpu-tile-full200k]") {
    testBackend(Backend::GpuTileFull200k);
}
#endif
