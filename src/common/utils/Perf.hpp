#ifndef PERF_HPP_
#define PERF_HPP_

#include <cstddef>

class Perf {
  private:
    unsigned long tStart;
    unsigned long tStop;

  public:
    Perf();
    Perf(const Perf &p);
    Perf(float ms);
    virtual ~Perf();

    void start();
    void stop();
    void reset();

    float getElapsedTime();                                                // ms
    float getGflops(float flops);                                          // Gflops/s
    float getFPS(const size_t nFrames = 1);                                // frames per second
    float getMemoryBandwidth(unsigned long memops, unsigned short nBytes); // Go/s

    Perf operator+(const Perf &p);
    Perf operator+=(const Perf &p);

  protected:
    static unsigned long getTime();
};

#endif /* PERF_HPP_ */
