#ifndef PERF_HPP_
#define PERF_HPP_

#include <cstddef>
#include <cstdint>

class Perf {
  private:
    std::uint64_t tStart;
    std::uint64_t tStop;

  public:
    Perf();
    Perf(const Perf &p);
    Perf(double ms);
    virtual ~Perf();

    void start();
    void stop();
    void reset();

    double getElapsedTime();                                                // ms
    double getGflops(double flops);                                          // Gflops/s
    double getFPS(const size_t nFrames = 1);                                // frames per second
    double getMemoryBandwidth(unsigned long memops, unsigned short nBytes); // Go/s

    Perf operator+(const Perf &p);
    Perf operator+=(const Perf &p);

  protected:
    static std::uint64_t getTime();
};

#endif /* PERF_HPP_ */
