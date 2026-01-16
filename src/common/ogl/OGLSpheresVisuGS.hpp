#ifdef VISU
#ifndef OGL_SPHERES_VISU_GS_HPP_
#define OGL_SPHERES_VISU_GS_HPP_

#include <map>
#include <string>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include "OGLSpheresVisu.hpp"

template <typename T> class OGLSpheresVisuGS : public OGLSpheresVisu<T> {
  public:
    OGLSpheresVisuGS(const std::string winName, const int winWidth, const int winHeight, const T *positionsX,
                     const T *positionsY, const T *positionsZ, const T *velocitiesX, const T *velocitiesY,
                     const T *velocitiesZ, const T *radius, const unsigned long nSpheres, const bool color = false);

    virtual ~OGLSpheresVisuGS();

    void refreshDisplay();
};

#endif /* OGL_SPHERES_VISU_GS_HPP_ */
#endif
