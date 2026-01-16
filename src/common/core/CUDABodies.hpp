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
    T* ax; /*!< Array of accelerations x. */
    T* ay; /*!< Array of accelerations y. */
    T* az; /*!< Array of accelerations z. */
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

  public:
    CUDABodies(const unsigned long n, const std::string &scheme = "galaxy", const unsigned long randInit = 0);

    const devDataSoA_t<T> &getDevDataSoA() const;

    virtual const dataSoA_t<T> &getDataSoA() const;
    virtual const std::vector<dataAoS_t<T>> &getDataAoS() const;

    virtual void updatePositionsAndVelocitiesOnDevice(const devAccSoA_t<T> &devAccelerations, T &dt);
    virtual void updatePositionsAndVelocities(const accSoA_t<T> &accelerations, T &dt);

    virtual void updatePositionsAndVelocities(const std::vector<accAoS_t<T>> &accelerations, T &dt);


    /*!
     *  \brief Initialized bodies like in a Galaxy with random generation. On device
     *  \warning Not implemented yet
     *
     *  \param randInit : Initialization number for random generation.
     */
    void initGalaxyOnDevice(const unsigned long randInit = 0);

    /*!
     *  \brief Initialized bodies randomly on device
     *  \warning Not implemented yet
     *
     *  \param randInit : Initialization number for random generation.
     */
    void initRandomlyOnDevice(const unsigned long randInit = 0);

    virtual ~CUDABodies();

  protected:
    /*!
     *  \brief Allocation of buffers on device
     */
    void allocateBuffersOnDevice();

    void memcpyBuffersOnDevice();
};