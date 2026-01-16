#ifdef VISU
#ifndef OGL_SPHERES_VISU_HPP_
#define OGL_SPHERES_VISU_HPP_

#include <map>
#include <string>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include "SpheresVisu.hpp"

#include "OGLControl.hpp"

template <typename T> class OGLSpheresVisu : public SpheresVisu {
  protected:
    GLFWwindow *window;

    const T *positionsX;
    float *positionsXBuffer;
    const T *positionsY;
    float *positionsYBuffer;
    const T *positionsZ;
    float *positionsZBuffer;
    const T *velocitiesX;
    float *velocitiesXBuffer;
    const T *velocitiesY;
    float *velocitiesYBuffer;
    const T *velocitiesZ;
    float *velocitiesZBuffer;
    const T *radius;
    float *radiusBuffer;
    float *colorBuffer;

    const unsigned long nSpheres;

    GLuint vertexArrayRef;
    GLuint positionBufferRef[3];
    GLuint accelerationBufferRef[3];
    GLuint radiusBufferRef;
    GLuint colorBufferRef;
    GLuint mvpRef;
    GLuint shaderProgramRef;

    glm::mat4 mvp;

    OGLControl *control;

    const bool color;

  protected:
    OGLSpheresVisu(const std::string winName, const int winWidth, const int winHeight, const T *positionsX,
                   const T *positionsY, const T *positionsZ, const T *velocitiesX, const T *velocitiesY,
                   const T *velocitiesZ, const T *radius, const unsigned long nSpheres, const bool color = false);
    OGLSpheresVisu();

  public:
    virtual ~OGLSpheresVisu();
    virtual void refreshDisplay() = 0;
    bool windowShouldClose();
    bool pressedSpaceBar();
    bool pressedPageUp();
    bool pressedPageDown();

  protected:
    bool compileShaders(const std::vector<GLenum> shadersType, const std::vector<std::string> shadersFiles);
    void updatePositions();
};

#endif /* OGL_SPHERES_VISU_HPP_ */
#endif
