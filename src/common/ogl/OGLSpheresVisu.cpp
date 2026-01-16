#ifdef VISU
#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_RADIANS
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include "OGLSpheresVisu.hpp"
#include "OGLTools.hpp"

template <typename T>
OGLSpheresVisu<T>::OGLSpheresVisu(const std::string winName, const int winWidth, const int winHeight,
                                  const T *positionsX, const T *positionsY, const T *positionsZ, const T *velocitiesX,
                                  const T *velocitiesY, const T *velocitiesZ, const T *radius,
                                  const unsigned long nSpheres, const bool color)
    : SpheresVisu(), window(NULL), positionsX(positionsX), positionsXBuffer(NULL), positionsY(positionsY),
      positionsYBuffer(NULL), positionsZ(positionsZ), positionsZBuffer(NULL), velocitiesX(velocitiesX),
      velocitiesXBuffer(NULL), velocitiesY(velocitiesY), velocitiesYBuffer(NULL), velocitiesZ(velocitiesZ),
      velocitiesZBuffer(NULL), radius(radius), radiusBuffer(NULL), colorBuffer(NULL), nSpheres(nSpheres),
      vertexArrayRef((GLuint)0), positionBufferRef{(GLuint)0, (GLuint)0, (GLuint)0},
      accelerationBufferRef{(GLuint)0, (GLuint)0, (GLuint)0}, radiusBufferRef((GLuint)0), colorBufferRef((GLuint)0),
      mvpRef((GLuint)0), shaderProgramRef((GLuint)0), mvp(glm::mat4(1.0f)), control(NULL), color(color)
{
    assert(winWidth > 0);
    assert(winHeight > 0);
    assert(positionsX);
    assert(positionsY);
    assert(positionsZ);
    assert(radius);

    if (sizeof(T) == sizeof(float)) {
        this->positionsXBuffer = const_cast<float *>((float *)positionsX);   // TODO: do not use const_cast !
        this->positionsYBuffer = const_cast<float *>((float *)positionsY);   // TODO: do not use const_cast !
        this->positionsZBuffer = const_cast<float *>((float *)positionsZ);   // TODO: do not use const_cast !
        this->velocitiesXBuffer = const_cast<float *>((float *)velocitiesX); // TODO: do not use const_cast !
        this->velocitiesYBuffer = const_cast<float *>((float *)velocitiesY); // TODO: do not use const_cast !
        this->velocitiesZBuffer = const_cast<float *>((float *)velocitiesZ); // TODO: do not use const_cast !
        this->radiusBuffer = const_cast<float *>((float *)radius);           // TODO: do not use const_cast !
    }
    else {
        this->positionsXBuffer = new float[this->nSpheres];
        this->positionsYBuffer = new float[this->nSpheres];
        this->positionsZBuffer = new float[this->nSpheres];
        this->velocitiesXBuffer = new float[this->nSpheres];
        this->velocitiesYBuffer = new float[this->nSpheres];
        this->velocitiesZBuffer = new float[this->nSpheres];
        this->radiusBuffer = new float[this->nSpheres];

        for (unsigned long iVertex = 0; iVertex < this->nSpheres; iVertex++)
            this->radiusBuffer[iVertex] = (float)this->radius[iVertex];
    }
    this->colorBuffer = new float[this->nSpheres * 3];

    this->window = OGLTools::initAndMakeWindow(winWidth, winHeight, winName.c_str());

    if (this->window) {
        glGenVertexArrays(1, &(this->vertexArrayRef));
        glBindVertexArray(this->vertexArrayRef);

        // bind position and radius buffers to OpenGL system
        glGenBuffers(1, &this->positionBufferRef[0]); // can change over iterations, so binding is in refreshDisplay()
        glGenBuffers(1, &this->positionBufferRef[1]); // can change over iterations, so binding is in refreshDisplay()
        glGenBuffers(1, &this->positionBufferRef[2]); // can change over iterations, so binding is in refreshDisplay()
        if (this->velocitiesX != nullptr && this->color) {
            // glGenBuffers(
            //     1, &this->accelerationBufferRef[0]); // can change over iterations, so binding is in refreshDisplay()
            // glGenBuffers(
            //     1, &this->accelerationBufferRef[1]); // can change over iterations, so binding is in refreshDisplay()
            // glGenBuffers(
            //     1, &this->accelerationBufferRef[2]); // can change over iterations, so binding is in refreshDisplay()
            glGenBuffers(1, &(this->colorBufferRef));
        }
        glGenBuffers(1, &(this->radiusBufferRef));
        glBindBuffer(GL_ARRAY_BUFFER, this->radiusBufferRef);
        glBufferData(GL_ARRAY_BUFFER, this->nSpheres * sizeof(GLfloat), this->radiusBuffer, GL_STATIC_DRAW);

        // set background color to black
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        // Create a control object in order to use mouse and keyboard (move in space)
        this->control = new OGLControl(this->window);

        // Enable depth test
        glEnable(GL_DEPTH_TEST);

        // Accept fragment if it closer to the camera than the former one
        glDepthFunc(GL_LESS);
    }
}

template <typename T>
OGLSpheresVisu<T>::OGLSpheresVisu()
    : SpheresVisu(), window(NULL), positionsX(NULL), positionsXBuffer(NULL), positionsY(NULL), positionsYBuffer(NULL),
      positionsZ(NULL), positionsZBuffer(NULL), velocitiesX(NULL), velocitiesXBuffer(NULL), velocitiesY(NULL),
      velocitiesYBuffer(NULL), velocitiesZ(NULL), velocitiesZBuffer(NULL), radius(NULL), radiusBuffer(NULL),
      nSpheres(0), vertexArrayRef((GLuint)0), positionBufferRef{(GLuint)0, (GLuint)0, (GLuint)0},
      accelerationBufferRef{(GLuint)0, (GLuint)0, (GLuint)0}, radiusBufferRef((GLuint)0), colorBufferRef((GLuint)0),
      mvpRef((GLuint)0), shaderProgramRef((GLuint)0), mvp(glm::mat4(1.0f)), control(NULL), color(false)
{
}

template <typename T> OGLSpheresVisu<T>::~OGLSpheresVisu()
{
    if (this->window)
        glfwDestroyWindow(this->window);

    if (this->control)
        delete this->control;

    if (sizeof(T) != sizeof(float)) {
        if (this->positionsXBuffer != nullptr) {
            delete[] this->positionsXBuffer;
            this->positionsXBuffer = nullptr;
        }
        if (this->positionsYBuffer != nullptr) {
            delete[] this->positionsYBuffer;
            this->positionsYBuffer = nullptr;
        }
        if (this->positionsZBuffer != nullptr) {
            delete[] this->positionsZBuffer;
            this->positionsZBuffer = nullptr;
        }
        if (this->velocitiesXBuffer != nullptr) {
            delete[] this->velocitiesXBuffer;
            this->velocitiesXBuffer = nullptr;
        }
        if (this->velocitiesYBuffer != nullptr) {
            delete[] this->velocitiesYBuffer;
            this->velocitiesYBuffer = nullptr;
        }
        if (this->velocitiesZBuffer != nullptr) {
            delete[] this->velocitiesZBuffer;
            this->velocitiesZBuffer = nullptr;
        }
        if (this->radiusBuffer != nullptr) {
            delete[] this->radiusBuffer;
            this->radiusBuffer = nullptr;
        }
        if (this->colorBuffer != nullptr) {
            delete[] this->colorBuffer;
            this->colorBuffer = nullptr;
        }
    }
}

// TODO: use map instead of two vectors ;-)
template <typename T>
bool OGLSpheresVisu<T>::compileShaders(const std::vector<GLenum> shadersType,
                                       const std::vector<std::string> shadersFiles)
{
    assert(shadersType.size() == shadersFiles.size());

    bool isFine = true;
    std::vector<GLuint> shaders;

    // load and compile shader programs
    for (int iShader = 0; iShader < (int)shadersType.size(); iShader++) {
        GLuint shader = OGLTools::loadShaderFromFile(shadersType[iShader], shadersFiles[iShader]);
        if (shader == 0)
            isFine = false;
        shaders.push_back(shader);
    }

    // link shader program
    if ((unsigned)(this->shaderProgramRef = OGLTools::linkShaders(shaders)) == 0)
        isFine = false;

    // ProjectionMatrix * ViewMatrix * ModelMatrix => MVP pattern (Model = identity here)
    // Get a handle for our "MVP" uniform
    this->mvpRef = glGetUniformLocation(this->shaderProgramRef, "MVP");

    for (int iShader = 0; iShader < (int)shaders.size(); iShader++)
        glDeleteShader(shaders[iShader]);

    return isFine;
}

template <typename T> void OGLSpheresVisu<T>::updatePositions()
{
    // convert positions in float (if necessary)
    if (sizeof(T) != sizeof(float)) {
        for (unsigned long iVertex = 0; iVertex < this->nSpheres; iVertex++) {
            this->positionsXBuffer[iVertex] = (float)this->positionsX[iVertex];
            this->positionsYBuffer[iVertex] = (float)this->positionsY[iVertex];
            this->positionsZBuffer[iVertex] = (float)this->positionsZ[iVertex];
        }
        if (this->velocitiesX != nullptr)
            for (unsigned long iVertex = 0; iVertex < this->nSpheres; iVertex++)
                this->velocitiesXBuffer[iVertex] = (float)this->velocitiesX[iVertex];
        if (this->velocitiesY != nullptr)
            for (unsigned long iVertex = 0; iVertex < this->nSpheres; iVertex++)
                this->velocitiesYBuffer[iVertex] = (float)this->velocitiesY[iVertex];
        if (this->velocitiesZ != nullptr)
            for (unsigned long iVertex = 0; iVertex < this->nSpheres; iVertex++)
                this->velocitiesZBuffer[iVertex] = (float)this->velocitiesZ[iVertex];
    }

    // bind position buffers to GPU
    glBindBuffer(GL_ARRAY_BUFFER, this->positionBufferRef[0]);
    glBufferData(GL_ARRAY_BUFFER, this->nSpheres * sizeof(GLfloat), this->positionsXBuffer, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, this->positionBufferRef[1]);
    glBufferData(GL_ARRAY_BUFFER, this->nSpheres * sizeof(GLfloat), this->positionsYBuffer, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, this->positionBufferRef[2]);
    glBufferData(GL_ARRAY_BUFFER, this->nSpheres * sizeof(GLfloat), this->positionsZBuffer, GL_STATIC_DRAW);

    if (this->velocitiesX != nullptr && this->color) {
        // glBindBuffer(GL_ARRAY_BUFFER, this->accelerationBufferRef[0]);
        // glBufferData(GL_ARRAY_BUFFER, this->nSpheres * sizeof(GLfloat), this->velocitiesXBuffer, GL_STATIC_DRAW);
        // glBindBuffer(GL_ARRAY_BUFFER, this->accelerationBufferRef[1]);
        // glBufferData(GL_ARRAY_BUFFER, this->nSpheres * sizeof(GLfloat), this->velocitiesYBuffer, GL_STATIC_DRAW);
        // glBindBuffer(GL_ARRAY_BUFFER, this->accelerationBufferRef[2]);
        // glBufferData(GL_ARRAY_BUFFER, this->nSpheres * sizeof(GLfloat), this->velocitiesZBuffer, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, this->colorBufferRef);
        glBufferData(GL_ARRAY_BUFFER, this->nSpheres * 3 * sizeof(GLfloat), this->colorBuffer, GL_STATIC_DRAW);
    }
}

template <typename T> bool OGLSpheresVisu<T>::windowShouldClose()
{
    if (this->window)
        return (bool)glfwWindowShouldClose(this->window);
    else
        return false;
}

template <typename T> bool OGLSpheresVisu<T>::pressedSpaceBar() { return glfwGetKey(this->window, GLFW_KEY_SPACE); }

template <typename T> bool OGLSpheresVisu<T>::pressedPageUp() { return glfwGetKey(this->window, GLFW_KEY_PAGE_UP); }

template <typename T> bool OGLSpheresVisu<T>::pressedPageDown() { return glfwGetKey(this->window, GLFW_KEY_PAGE_DOWN); }

// ==================================================================================== explicit template instantiation
template class OGLSpheresVisu<double>;
template class OGLSpheresVisu<float>;
// ==================================================================================== explicit template instantiation
#endif
