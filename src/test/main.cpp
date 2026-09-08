#define CATCH_CONFIG_RUNNER

#include <catch.hpp>

#include <cstdlib>
#include <exception>
#include <iostream>

#ifdef USE_MPI
#include "MultiGpuRuntime.hpp"
#endif

int main(int argc, char *argv[])
{
#ifdef USE_MPI
    try {
        const bool diagnostics = std::getenv("MURB_MPI_DIAGNOSTICS") != nullptr;
        murb::MultiGpuRuntime runtime(argc, argv, false, diagnostics);
        const int result = Catch::Session().run(argc, argv);
        runtime.finalize();
        return result;
    } catch (const std::exception& error) {
        std::cerr << "murb-test: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
#else
    int result = Catch::Session().run(argc, argv);

    return result;
#endif
}
