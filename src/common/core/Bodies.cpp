#include "Bodies.hpp"

#include <mipp.h>
#include <sys/stat.h>

#include <cassert>
#include <cmath>
#include <limits>
#include <string>

#include "../utils/Perf.hpp"

template <typename T>
Bodies<T>::Bodies(const unsigned long n, const std::string &scheme, const unsigned long randInit)
    : n(n), padding(0), allocatedBytes(0)
{
    assert(n > 0);
    if (scheme == "galaxy")
        this->initGalaxy(randInit);
    else if (scheme == "random")
        this->initRandomly(randInit);
    else {
        std::cout << "(EE) `scheme` must be either `galaxy` or `random`." << std::endl;
        std::exit(-1);
    }
}

template <typename T> void Bodies<T>::allocateBuffers()
{
    this->dataSoA.m.resize(this->n + this->padding);
    this->dataSoA.r.resize(this->n + this->padding);
    this->dataSoA.qx.resize(this->n + this->padding);
    this->dataSoA.qy.resize(this->n + this->padding);
    this->dataSoA.qz.resize(this->n + this->padding);
    this->dataSoA.vx.resize(this->n + this->padding);
    this->dataSoA.vy.resize(this->n + this->padding);
    this->dataSoA.vz.resize(this->n + this->padding);

    this->dataAoS.resize(this->n + this->padding);

    this->allocatedBytes = (this->n + this->padding) * sizeof(T) * 8 * 2;
}

template <typename T> const unsigned long Bodies<T>::getN() const { return this->n; }

template <typename T> const unsigned short Bodies<T>::getPadding() const { return this->padding; }

template <typename T> const dataSoA_t<T> &Bodies<T>::getDataSoA() const { return this->dataSoA; }

template <typename T> const std::vector<dataAoS_t<T>> &Bodies<T>::getDataAoS() const { return this->dataAoS; }

template <typename T> const float Bodies<T>::getAllocatedBytes() const { return this->allocatedBytes; }

template <typename T>
void Bodies<T>::setBody(const unsigned long &iBody, const T &mi, const T &ri, const T &qix, const T &qiy, const T &qiz,
                        const T &vix, const T &viy, const T &viz)
{
    // SoA
    this->dataSoA.m[iBody] = mi;
    this->dataSoA.r[iBody] = ri;
    this->dataSoA.qx[iBody] = qix;
    this->dataSoA.qy[iBody] = qiy;
    this->dataSoA.qz[iBody] = qiz;
    this->dataSoA.vx[iBody] = vix;
    this->dataSoA.vy[iBody] = viy;
    this->dataSoA.vz[iBody] = viz;
    // AoS
    this->dataAoS[iBody].m = mi;
    this->dataAoS[iBody].r = ri;
    this->dataAoS[iBody].qx = qix;
    this->dataAoS[iBody].qy = qiy;
    this->dataAoS[iBody].qz = qiz;
    this->dataAoS[iBody].vx = vix;
    this->dataAoS[iBody].vy = viy;
    this->dataAoS[iBody].vz = viz;
}

/* create a galaxy... */
template <typename T> void Bodies<T>::initGalaxy(const unsigned long randInit)
{
    const auto nVecs = ceil((T)this->n / (T)mipp::N<T>());
    this->padding = (nVecs * mipp::N<T>()) - this->n;

    this->allocateBuffers();

    srand(randInit);
    for (unsigned long iBody = 0; iBody < this->n; iBody++) {
        // srand(iBody);
        T mi, ri, qix, qiy, qiz, vix, viy, viz;

        if (iBody == 0) {
            mi = 2.0e24;
            ri = 0.0e6;
            qix = 0.0;
            qiy = 0.0;
            qiz = 0.0;
            vix = 0;
            viy = 0;
            viz = 0;
        }
        else {
            mi = ((rand() / (T)RAND_MAX) * 5e20);
            ri = mi * 2.5e-15;

            T horizontalAngle = ((RAND_MAX - rand()) / (T)(RAND_MAX)) * 2.0 * M_PI;
            T verticalAngle = ((RAND_MAX - rand()) / (T)(RAND_MAX)) * 2.0 * M_PI;
            T distToCenter = ((RAND_MAX - rand()) / (T)(RAND_MAX)) * 1.0e8 + 1.0e8;

            qix = std::cos(verticalAngle) * std::sin(horizontalAngle) * distToCenter;
            qiy = std::sin(verticalAngle) * distToCenter;
            qiz = std::cos(verticalAngle) * std::cos(horizontalAngle) * distToCenter;

            vix = qiy * 4.0e-6;
            viy = -qix * 4.0e-6;
            viz = 0.0e2;
        }

        this->setBody(iBody, mi, ri, qix, qiy, qiz, vix, viy, viz);
    }

    // fill the bodies in the padding zone
    for (unsigned long iBody = this->n; iBody < this->n + this->padding; iBody++) {
        T qix, qiy, qiz, vix, viy, viz;

        qix = ((rand() - RAND_MAX / 2) / (T)(RAND_MAX / 2)) * (5.0e8 * 1.33);
        qiy = ((rand() - RAND_MAX / 2) / (T)(RAND_MAX / 2)) * 5.0e8;
        qiz = ((rand() - RAND_MAX / 2) / (T)(RAND_MAX / 2)) * 5.0e8 - 10.0e8;

        vix = ((rand() - RAND_MAX / 2) / (T)(RAND_MAX / 2)) * 1.0e2;
        viy = ((rand() - RAND_MAX / 2) / (T)(RAND_MAX / 2)) * 1.0e2;
        viz = ((rand() - RAND_MAX / 2) / (T)(RAND_MAX / 2)) * 1.0e2;

        this->setBody(iBody, 0, 0, qix, qiy, qiz, vix, viy, viz);
    }
}

/* real random */
template <typename T> void Bodies<T>::initRandomly(const unsigned long randInit)
{
    const auto nVecs = ceil((T)this->n / (T)mipp::N<T>());
    this->padding = (nVecs * mipp::N<T>()) - this->n;

    this->allocateBuffers();

    srand(randInit);
    for (unsigned long iBody = 0; iBody < this->n; iBody++) {
        T mi, ri, qix, qiy, qiz, vix, viy, viz;

        mi = ((rand() / (T)RAND_MAX) * 5.0e21);

        ri = mi * 0.5e-14;

        qix = ((rand() - RAND_MAX / 2) / (T)(RAND_MAX / 2)) * (5.0e8 * 1.33);
        qiy = ((rand() - RAND_MAX / 2) / (T)(RAND_MAX / 2)) * 5.0e8;
        qiz = ((rand() - RAND_MAX / 2) / (T)(RAND_MAX / 2)) * 5.0e8 - 10.0e8;

        vix = ((rand() - RAND_MAX / 2) / (T)(RAND_MAX / 2)) * 1.0e2;
        viy = ((rand() - RAND_MAX / 2) / (T)(RAND_MAX / 2)) * 1.0e2;
        viz = ((rand() - RAND_MAX / 2) / (T)(RAND_MAX / 2)) * 1.0e2;

        this->setBody(iBody, mi, ri, qix, qiy, qiz, vix, viy, viz);
    }

    // fill the bodies in the padding zone
    for (unsigned long iBody = this->n; iBody < this->n + this->padding; iBody++) {
        T qix, qiy, qiz, vix, viy, viz;

        qix = ((rand() - RAND_MAX / 2) / (T)(RAND_MAX / 2)) * (5.0e8 * 1.33);
        qiy = ((rand() - RAND_MAX / 2) / (T)(RAND_MAX / 2)) * 5.0e8;
        qiz = ((rand() - RAND_MAX / 2) / (T)(RAND_MAX / 2)) * 5.0e8 - 10.0e8;

        vix = ((rand() - RAND_MAX / 2) / (T)(RAND_MAX / 2)) * 1.0e2;
        viy = ((rand() - RAND_MAX / 2) / (T)(RAND_MAX / 2)) * 1.0e2;
        viz = ((rand() - RAND_MAX / 2) / (T)(RAND_MAX / 2)) * 1.0e2;

        this->setBody(iBody, 0, 0, qix, qiy, qiz, vix, viy, viz);
    }
}

template <typename T>
void Bodies<T>::updatePositionAndVelocity(const unsigned long iBody, const T mi, const T ri, const T qix, const T qiy,
                                          const T qiz, const T vix, const T viy, const T viz, const T aix, const T aiy,
                                          const T aiz, T &dt)
{
    // flops = 18
    T aixDt = aix * dt;
    T aiyDt = aiy * dt;
    T aizSt = aiz * dt;

    T qixNew = qix + (vix + aixDt * 0.5) * dt;
    T qiyNew = qiy + (viy + aiyDt * 0.5) * dt;
    T qizNew = qiz + (viz + aizSt * 0.5) * dt;

    T vixNew = vix + aixDt;
    T viyNew = viy + aiyDt;
    T vizNew = viz + aizSt;

    this->setBody(iBody, mi, ri, qixNew, qiyNew, qizNew, vixNew, viyNew, vizNew);
}

template <typename T> void Bodies<T>::updatePositionsAndVelocities(const accSoA_t<T> &accelerations, T &dt)
{
    // flops = n * 18
    for (unsigned long iBody = 0; iBody < this->n; iBody++)
        updatePositionAndVelocity(iBody, this->dataSoA.m[iBody], this->dataSoA.r[iBody], this->dataSoA.qx[iBody],
                                  this->dataSoA.qy[iBody], this->dataSoA.qz[iBody], this->dataSoA.vx[iBody],
                                  this->dataSoA.vy[iBody], this->dataSoA.vz[iBody], accelerations.ax[iBody],
                                  accelerations.ay[iBody], accelerations.az[iBody], dt);
}

template <typename T> void Bodies<T>::updatePositionsAndVelocities(const std::vector<accAoS_t<T>> &accelerations, T &dt)
{
    // flops = n * 18
    for (unsigned long iBody = 0; iBody < this->n; iBody++)
        updatePositionAndVelocity(iBody, this->dataSoA.m[iBody], this->dataSoA.r[iBody], this->dataSoA.qx[iBody],
                                  this->dataSoA.qy[iBody], this->dataSoA.qz[iBody], this->dataSoA.vx[iBody],
                                  this->dataSoA.vy[iBody], this->dataSoA.vz[iBody], accelerations[iBody].ax,
                                  accelerations[iBody].ay, accelerations[iBody].az, dt);
}

// ==================================================================================== explicit template instantiation
template class Bodies<double>;
template class Bodies<float>;
// ==================================================================================== explicit template instantiation
