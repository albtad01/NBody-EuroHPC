#include "Perf.hpp"

#include <chrono>

Perf::Perf() : tStart(0), tStop(0) {}

Perf::Perf(const Perf &p) : tStart(p.tStart), tStop(p.tStop) {}

Perf::Perf(double ms) : tStart(0), tStop(ms > 0 ? static_cast<std::uint64_t>(ms * 1e6) : 0) {}

Perf::~Perf() {}

void Perf::start() { this->tStart = Perf::getTime(); }

void Perf::stop() { this->tStop = Perf::getTime(); }

void Perf::reset()
{
    this->tStart = 0;
    this->tStop = 0;
}

double Perf::getElapsedTime() { return (this->tStop - this->tStart) / 1e6; }

double Perf::getGflops(double flops) { return getElapsedTime() > 0 ? flops / (getElapsedTime() * 1e6) : 0.0; }

double Perf::getFPS(const size_t nFrames) { return getElapsedTime() > 0 ? nFrames * 1000.0 / getElapsedTime() : 0.0; }

double Perf::getMemoryBandwidth(unsigned long memops, unsigned short nBytes)
{
    return getElapsedTime() > 0 ? (static_cast<double>(memops) * nBytes * (1000 / getElapsedTime())) / 1024.0 / 1024.0 / 1024.0 : 0.0;
}

Perf Perf::operator+(const Perf &p)
{
    Perf pAdd;
    pAdd.tStop = (p.tStop - p.tStart) + (this->tStop - this->tStart);
    return pAdd;
}

Perf Perf::operator+=(const Perf &p)
{
    this->tStop += p.tStop - p.tStart;
    return (*this);
}

std::uint64_t Perf::getTime()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
