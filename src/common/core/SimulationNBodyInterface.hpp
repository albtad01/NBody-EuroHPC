#ifndef SIMULATION_N_BODY_INTERFACE_HPP_
#define SIMULATION_N_BODY_INTERFACE_HPP_

#include <string>
#include <concepts>
#include <memory>

#include "Bodies.hpp"
#include "BodiesAllocator.hpp"

/*!
 * \class  SimulationNBodyInterface
 * \brief  This is the main simulation class, it describes the main methods to implement in extended classes.
 */
template <typename T>
class SimulationNBodyInterface {
  protected:
    const T G = 6.67384e-11f; /*!< The gravitational constant in m^3.kg^-1.s^-2. */
    const BodiesAllocatorInterface<T>& allocator;
    std::shared_ptr<Bodies<T>> bodies;         /*!< Bodies object, represent all the bodies available in space. */
    T dt;                     /*!< Time step value. */
    T soft;                   /*!< Softening factor value. */
    T flopsPerIte;            /*!< Number of floating-point operations per iteration. */
    T allocatedBytes;         /*!< Number of allocated bytes. */

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
    SimulationNBodyInterface(const BodiesAllocatorInterface<T>& allocator, const T soft = 0.035f);

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
    const std::shared_ptr<Bodies<T>> &getBodies() const;

    /*!
     *  \brief dt setter.
     *
     *  \param dtVal : Constant time step value.
     */
    void setDt(T dtVal);

    /*!
     *  \brief Time step getter.
     *
     *  \return Time step value.
     */
    const T getDt() const;

    /*!
     *  \brief Flops per iteration getter.
     *
     *  \return Flops per iteration.
     */
    const T getFlopsPerIte() const;

    /*!
     *  \brief Allocated bytes getter.
     *
     *  \return Number of allocated bytes.
     */
    const T getAllocatedBytes() const;
};


#endif /* SIMULATION_N_BODY_INTERFACE_HPP_ */
