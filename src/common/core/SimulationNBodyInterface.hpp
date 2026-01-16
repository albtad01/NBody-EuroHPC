#ifndef SIMULATION_N_BODY_INTERFACE_HPP_
#define SIMULATION_N_BODY_INTERFACE_HPP_

#include <string>

#include "Bodies.hpp"

/*!
 * \class  SimulationNBodyInterface
 * \brief  This is the main simulation class, it describes the main methods to implement in extended classes.
 */
class SimulationNBodyInterface {
  protected:
    const float G = 6.67384e-11f; /*!< The gravitational constant in m^3.kg^-1.s^-2. */
    Bodies<float> bodies;         /*!< Bodies object, represent all the bodies available in space. */
    float dt;                     /*!< Time step value. */
    float soft;                   /*!< Softening factor value. */
    float flopsPerIte;            /*!< Number of floating-point operations per iteration. */
    float allocatedBytes;         /*!< Number of allocated bytes. */

  protected:
    /*!
     *  \brief Constructor.
     *
     *  n-body simulation interface.
     *
     *  \param nBodies   : Number of bodies.
     *  \param scheme    : `galaxy` or `random`
     *  \param soft      : Softening factor value.
     *  \param randInit  : PNRG seed.
     */
    SimulationNBodyInterface(const unsigned long nBodies, const std::string &scheme = "galaxy",
                             const float soft = 0.035f, const unsigned long randInit = 0);

  public:
    /*!
     *  \brief Main compute method.
     *
     *  Compute one iteration of the simulation.
     */
    virtual void computeOneIteration() = 0;

    /*!
     *  \brief Destructor.
     *
     *  SimulationNBodyInterface destructor.
     */
    virtual ~SimulationNBodyInterface() = default;

    /*!
     *  \brief Bodies getter.
     *
     *  \return Bodies class.
     */
    const Bodies<float> &getBodies() const;

    /*!
     *  \brief dt setter.
     *
     *  \param dtVal : Constant time step value.
     */
    void setDt(float dtVal);

    /*!
     *  \brief Time step getter.
     *
     *  \return Time step value.
     */
    const float getDt() const;

    /*!
     *  \brief Flops per iteration getter.
     *
     *  \return Flops per iteration.
     */
    const float getFlopsPerIte() const;

    /*!
     *  \brief Allocated bytes getter.
     *
     *  \return Number of allocated bytes.
     */
    const float getAllocatedBytes() const;
};

#endif /* SIMULATION_N_BODY_INTERFACE_HPP_ */
