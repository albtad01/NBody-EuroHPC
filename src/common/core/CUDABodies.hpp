#ifndef CUDA_BODIES_HPP_
#define CUDA_BODIES_HPP_

#include "core/Bodies.hpp"

template <typename T> struct devDataSoA_t {
    T* qx; /*!< Array of positions x. */
    T* qy; /*!< Array of positions y. */
    T* qz; /*!< Array of positions z. */
    T* vx; /*!< Array of velocities x. */
    T* vy; /*!< Array of velocities y. */
    T* vz; /*!< Array of velocities z. */
    T* m;  /*!< Array of masses. */
    T* r;  /*!< Array of radiuses. */
};


template <typename T> struct devAccSoA_t {
    T* x; /*!< Array of accelerations x. */
    T* y; /*!< Array of accelerations y. */
    T* z; /*!< Array of accelerations z. */
};

template <typename T> class CUDABodies : public Bodies<T> {
  protected:
    // =================================== INHERITED =====================================
    // unsigned long n;                   /*!< Number of bodies. */
    // dataSoA_t<T> dataSoA;              /*!< Structure of arrays of bodies data. */
    // std::vector<dataAoS_t<T>> dataAoS; /*!< Array of structures of bodies data. */
    // unsigned short padding;            /*!< Number of fictional bodies to fill the last vector. */
    // float allocatedBytes;              /*!< Number of allocated bytes. */

    // =============================== DEVICE ATTRIBUTES =================================
    devDataSoA_t<T> devDataSoA;
    devAccSoA_t<T> devIntermVelocities;
    devAccSoA_t<T> devNextPositions;
    mutable bool dataOnCPU = false;

  public:
    CUDABodies(const unsigned long n, const std::string &scheme = "galaxy", const unsigned long randInit = 0);

    const devAccSoA_t<T>& getDevPositionsBuffer() const;

    const devDataSoA_t<T> &getDevDataSoA() const;
    void invalidateDataSoA();
    virtual const dataSoA_t<T> &getDataSoA() const;
    virtual const std::vector<dataAoS_t<T>> &getDataAoS() const;

    virtual void updatePositionsAndVelocitiesLeapfrogOnDevice(const devAccSoA_t<T> &devAccelerations, 
                                                              T &dt, int iteration, int totalIterations);
    virtual void updatePositionsAndVelocitiesOnDevice(const devAccSoA_t<T> &devAccelerations, T &dt);
    virtual void updatePositionsAndVelocities(const accSoA_t<T> &accelerations, T &dt);

    virtual void updatePositionsAndVelocities(const std::vector<accAoS_t<T>> &accelerations, T &dt);

    virtual ~CUDABodies();

  protected:
    /*!
     *  \brief Allocation of buffers on device
     */
    void allocateBuffersOnDevice();

    void memcpyBuffersOnDevice();
};
#endif