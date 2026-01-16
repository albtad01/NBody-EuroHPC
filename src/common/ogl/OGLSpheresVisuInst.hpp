#ifdef VISU
#ifndef OGL_SPHERES_VISU_INST_HPP_
#define OGL_SPHERES_VISU_INST_HPP_

#include <map>
#include <string>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include "OGLSpheresVisu.hpp"

template <typename T> class OGLSpheresVisuInst : public OGLSpheresVisu<T> {
  private:
    const float PI = 3.1415926;
    // const unsigned long  nPointsPerCircle = 22;
    const unsigned long nPointsPerCircle = 8;
    unsigned long vertexModelSize;
    GLfloat *vertexModel;
    GLuint modelBufferRef;

  public:
    OGLSpheresVisuInst(const std::string winName, const int winWidth, const int winHeight, const T *positionsX,
                       const T *positionsY, const T *positionsZ, const T *velocitiesX, const T *velocitiesY,
                       const T *velocitiesZ, const T *radius, const unsigned long nSpheres, const bool color = false);

    virtual ~OGLSpheresVisuInst();
    void refreshDisplay();
};

#endif /* OGL_SPHERES_VISU_INST_HPP_ */
#endif
