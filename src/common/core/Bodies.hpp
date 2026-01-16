#ifndef BODIES_HPP_
#define BODIES_HPP_

#include <string>
#include <vector>

/*!
 * \struct dataSoA_t
 * \brief  Structure of arrays.
 *
 * \tparam T : Type.
 *
 * The dataSoA_t structure represent the characteristics of the bodies.
 */
template <typename T> struct dataSoA_t {
    std::vector<T> qx; /*!< Array of positions x. */
    std::vector<T> qy; /*!< Array of positions y. */
    std::vector<T> qz; /*!< Array of positions z. */
    std::vector<T> vx; /*!< Array of velocities x. */
    std::vector<T> vy; /*!< Array of velocities y. */
    std::vector<T> vz; /*!< Array of velocities z. */
    std::vector<T> m;  /*!< Array of masses. */
    std::vector<T> r;  /*!< Array of radiuses. */
};

/*!
 * \struct dataAoS_t
 * \brief  Structure of body characteristics.
 *
 * \tparam T : Type.
 *
 * The characteristics of a body.
 */
template <typename T> struct dataAoS_t {
    T qx; /*!< Position x. */
    T qy; /*!< Position y. */
    T qz; /*!< Position z. */
    T vx; /*!< Velocity x. */
    T vy; /*!< Velocity y. */
    T vz; /*!< Velocity z. */
    T m;  /*!< Mass. */
    T r;  /*!< Radius. */
};

/*!
 * \struct accSoA_t
 * \brief  Structure of arrays.
 *
 * \tparam T : Type.
 *
 * The accSoA_t structure represent the accelerations of the bodies.
 */
template <typename T> struct accSoA_t {
    std::vector<T> ax; /*!< Array of accelerations x. */
    std::vector<T> ay; /*!< Array of accelerations y. */
    std::vector<T> az; /*!< Array of accelerations z. */
};

/*!
 * \struct accAoS_t
 * \brief  Structure of body acceleration.
 *
 * \tparam T : Type.
 *
 * The body acceleration.
 */
template <typename T> struct accAoS_t {
    T ax; /*!< Acceleration x. */
    T ay; /*!< Acceleration y. */
    T az; /*!< Acceleration z. */
};

/*!
 * \class  Bodies
 * \brief  Bodies class represents the physic data of each body (mass, radius, position and velocity).
 *
 * \tparam T : Float type.
 */
template <typename T> class Bodies {
  protected:
    unsigned long n;                   /*!< Number of bodies. */
    dataSoA_t<T> dataSoA;              /*!< Structure of arrays of bodies data. */
    std::vector<dataAoS_t<T>> dataAoS; /*!< Array of structures of bodies data. */
    unsigned short padding;            /*!< Number of fictional bodies to fill the last vector. */
    float allocatedBytes;              /*!< Number of allocated bytes. */

  public:
    /*!
     *  \brief Constructor.
     *
     *  Bodies constructor : generates random bodies in space.
     *
     *  \param n        : Number of bodies.
     *  \param scheme   : Type of initialization (galaxy or random).
     *  \param randInit : Initialization number for random generation.
     */
    Bodies(const unsigned long n, const std::string &scheme = "galaxy", const unsigned long randInit = 0);

    /*!
     *  \brief Destructor.
     *
     *  Bodies destructor.
     */
    virtual ~Bodies() = default;

    /*!
     *  \brief N getter.
     *
     *  \return The number of bodies.
     */
    const unsigned long getN() const;

    /*!
     *  \brief Padding getter.
     *
     *  \return The number of bodies in the padding zone.
     */
    const unsigned short getPadding() const;

    /*!
     *  \brief SoA data getter.
     *
     *  \return The characteristics of the bodies in SoA form.
     */
    const dataSoA_t<T> &getDataSoA() const;

    /*!
     *  \brief AoS data getter.
     *
     *  \return The characteristics of the bodies in AoS form.
     */
    const std::vector<dataAoS_t<T>> &getDataAoS() const;

    /*!
     *  \brief Allocated bytes getter.
     *
     *  \return The number of allocated bytes.
     */
    const float getAllocatedBytes() const;

    /*!
     *  \brief Update positions and velocities array.
     *
     *  \param accelerations : The array of accelerations needed to compute new positions and velocities (SoA).
     *  \param dt            : The time step value (required for time integration scheme).
     *
     *  Update positions and velocities, this is the time integration scheme to apply after each iteration.
     */
    void updatePositionsAndVelocities(const accSoA_t<T> &accelerations, T &dt);

    /*!
     *  \brief Update positions and velocities with time integration.
     *
     *  \param accelerations : The array of accelerations needed to compute new positions and velocities (AoS).
     *  \param dt            : The time step value (required for time integration scheme).
     *
     *  Update positions and velocities, this is the time integration scheme to apply after each iteration.
     */
    void updatePositionsAndVelocities(const std::vector<accAoS_t<T>> &accelerations, T &dt);

    /*!
     *  \brief Initialized bodies like in a Galaxy with random.
     *
     *  \param randInit : Initialization number for random generation.
     */
    void initGalaxy(const unsigned long randInit = 0);

    /*!
     *  \brief Initialized bodies randomly.
     *
     *  \param randInit : Initialization number for random generation.
     */
    void initRandomly(const unsigned long randInit = 0);

  protected:
    /*!
     *  \brief Update the position and the velocity of one body with time integration.
     *
     *  \param iBody : Body i id.
     *  \param mi    : Body i mass.
     *  \param ri    : Body i radius.
     *  \param qix   : Body i position x.
     *  \param qiy   : Body i position y.
     *  \param qiz   : Body i position z.
     *  \param vix   : Body i velocity x.
     *  \param viy   : Body i velocity y.
     *  \param viz   : Body i velocity z.
     *  \param aix   : Body i acceleration x.
     *  \param aiy   : Body i acceleration y.
     *  \param aiz   : Body i acceleration z.
     *  \param dt    : The time step value (required for time integration scheme).
     *
     *  This function is called by the `updatePositionsAndVelocities` methods.
     */
    void updatePositionAndVelocity(const unsigned long iBody, const T mi, const T ri, const T qix, const T qiy,
                                   const T qiz, const T vix, const T viy, const T viz, const T aix, const T aiy,
                                   const T aiz, T &dt);

    /*!
     *  \brief Body setter.
     *
     *  \param iBody : Body i id.
     *  \param mi    : Body i mass.
     *  \param ri    : Body i radius.
     *  \param qix   : Body i position x.
     *  \param qiy   : Body i position y.
     *  \param qiz   : Body i position z.
     *  \param vix   : Body i velocity x.
     *  \param viy   : Body i velocity y.
     *  \param viz   : Body i velocity z.
     */
    inline void setBody(const unsigned long &iBody, const T &mi, const T &ri, const T &qix, const T &qiy, const T &qiz,
                        const T &vix, const T &viy, const T &viz);

    /*!
     *  \brief Allocation of buffers.
     */
    void allocateBuffers();
};

#endif /* BODIES_HPP_ */
