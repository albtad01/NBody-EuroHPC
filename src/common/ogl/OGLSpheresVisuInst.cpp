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

#include "OGLSpheresVisuInst.hpp"
#include "OGLTools.hpp"

template <typename T>
OGLSpheresVisuInst<T>::OGLSpheresVisuInst(const std::string winName, const int winWidth, const int winHeight,
                                          const T *positionsX, const T *positionsY, const T *positionsZ,
                                          const T *velocitiesX, const T *velocitiesY, const T *velocitiesZ,
                                          const T *radius, const unsigned long nSpheres, const bool color)
    : OGLSpheresVisu<T>(winName, winWidth, winHeight, positionsX, positionsY, positionsZ, velocitiesX,
                        velocitiesX ? velocitiesY : nullptr, velocitiesX ? velocitiesZ : nullptr, radius, nSpheres,
                        color),
      vertexModelSize((this->nPointsPerCircle + 1) * ((this->nPointsPerCircle / 2) + 1) * 3), vertexModel(NULL),
      modelBufferRef((GLuint)0)
{
    // make sphere model
    this->vertexModel = new GLfloat[this->vertexModelSize];

    for (int j = 0; j <= (int)(this->nPointsPerCircle / 2); j++) {
        float horizontalAngle = (PI * 2.0 / this->nPointsPerCircle) * j;

        for (int i = 0; i <= (int)this->nPointsPerCircle; i++) {
            // Angle between each side in radiant
            float verticalAngle = (PI * 2.0 / this->nPointsPerCircle) * i + (3.14f / 2.0f);

            this->vertexModel[(i + j * (this->nPointsPerCircle + 1)) * 3 + 0] =
                cos(verticalAngle) * sin(horizontalAngle);
            this->vertexModel[(i + j * (this->nPointsPerCircle + 1)) * 3 + 1] = sin(verticalAngle);
            this->vertexModel[(i + j * (this->nPointsPerCircle + 1)) * 3 + 2] =
                cos(verticalAngle) * cos(horizontalAngle);
        }
    }

    if (this->window) {
        // bind vertex sphere model buffer to OpenGL system
        glGenBuffers(1, &(this->modelBufferRef));
        glBindBuffer(GL_ARRAY_BUFFER, this->modelBufferRef);
        glBufferData(GL_ARRAY_BUFFER, this->vertexModelSize * sizeof(GLfloat), this->vertexModel, GL_STATIC_DRAW);

        // specify shaders path and compile them
        std::vector<GLenum> shadersType(2);
        std::vector<std::string> shadersFiles(2);
        shadersType[0] = GL_VERTEX_SHADER;
        shadersFiles[0] = velocitiesX && color ? "../src/common/ogl/shaders/vertex130_color.glsl"
                                               : "../src/common/ogl/shaders/vertex130.glsl";
        shadersType[1] = GL_FRAGMENT_SHADER;
        shadersFiles[1] = velocitiesX && color ? "../src/common/ogl/shaders/fragment130_color.glsl"
                                               : "../src/common/ogl/shaders/fragment130.glsl";

        this->compileShaders(shadersType, shadersFiles);
    }
}

template <typename T> OGLSpheresVisuInst<T>::~OGLSpheresVisuInst()
{
    if (this->vertexModel)
        delete[] this->vertexModel;
}

template <typename T> void OGLSpheresVisuInst<T>::refreshDisplay()
{
    if (this->window) {
        this->updatePositions();

        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // use our shader program
        if (this->shaderProgramRef != 0)
            glUseProgram(this->shaderProgramRef);

        // 1rst attribute buffer : vertex model
        int iBufferIndex = 0;
        glEnableVertexAttribArray(iBufferIndex);
        glBindBuffer(GL_ARRAY_BUFFER, this->modelBufferRef);
        glVertexAttribPointer(
            iBufferIndex++, // attribute. No particular reason for 0, but must match the layout in the shader.
            3,              // size
            GL_FLOAT,       // type
            GL_FALSE,       // normalized?
            0,              // stride
            (void *)0       // array buffer offset
        );

        // 2nd attribute buffer : vertex positions
        for (int i = 0; i < 3; i++) {
            glEnableVertexAttribArray(iBufferIndex);
            glBindBuffer(GL_ARRAY_BUFFER, this->positionBufferRef[i]);
            glVertexAttribPointer(
                iBufferIndex++, // attribute. No particular reason for 0, but must match the layout in the shader.
                1,              // size
                GL_FLOAT,       // type
                GL_FALSE,       // normalized?
                0,              // stride
                (void *)0       // array buffer offset
            );
        }

        // 3rd attribute buffer : vertex velocities
        if (this->velocitiesX != nullptr)
            for (int i = 0; i < 3; i++) {
                glEnableVertexAttribArray(iBufferIndex);
                glBindBuffer(GL_ARRAY_BUFFER, this->accelerationBufferRef[i]);
                glVertexAttribPointer(
                    iBufferIndex++, // attribute. No particular reason for 0, but must match the layout in the shader.
                    1,              // size
                    GL_FLOAT,       // type
                    GL_FALSE,       // normalized?
                    0,              // stride
                    (void *)0       // array buffer offset
                );
            }

        // 4rd attribute buffer : radius
        glEnableVertexAttribArray(iBufferIndex);
        glBindBuffer(GL_ARRAY_BUFFER, this->radiusBufferRef);
        glVertexAttribPointer(
            iBufferIndex++, // attribute. No particular reason for 1, but must match the layout in the shader.
            1,              // size
            GL_FLOAT,       // type
            GL_FALSE,       // normalized?
            0,              // stride
            (void *)0       // array buffer offset
        );

        // Compute the MVP matrix from keyboard and mouse input
        this->mvp = this->control->computeViewAndProjectionMatricesFromInputs();

        // Send our transformation to the currently bound shader,
        // in the "MVP" uniform
        glUniformMatrix4fv(this->mvpRef, 1, GL_FALSE, &this->mvp[0][0]);

        // These functions are specific to glDrawArrays*Instanced*.
        // The first parameter is the attribute buffer we're talking about.
        // The second parameter is the "rate at which generic vertex attributes advance when rendering multiple
        // instances" http://www.opengl.org/sdk/docs/man/xhtml/glVertexAttribDivisor.xml
        glVertexAttribDivisor(0, 0); // model vertices
        glVertexAttribDivisor(1, 1); // positionsX : one per sphere (its center) -> 1
        glVertexAttribDivisor(2, 1); // positionsY : one per sphere (its center) -> 1
        glVertexAttribDivisor(3, 1); // positionsZ : one per sphere (its center) -> 1
        if (this->velocitiesX != nullptr) {
            glVertexAttribDivisor(4, 1); // velocitiesX : one per sphere (its center) -> 1
            glVertexAttribDivisor(5, 1); // velocitiesY : one per sphere (its center) -> 1
            glVertexAttribDivisor(6, 1); // velocitiesZ : one per sphere (its center) -> 1
            glVertexAttribDivisor(7, 1); // radius      : one per sphere -> 1
        }
        else
            glVertexAttribDivisor(4, 1); // radius     : one per sphere -> 1

        // Draw the particles !
        // This draws many times a small line_strip (which looks like a sphere).
        // This is equivalent to :
        //   for(i in NSpheres) : glDrawArrays(GL_TRIANGLE_STRIP, 0, 4),
        // but faster.
        glDrawArraysInstanced(GL_LINE_STRIP, 0, this->vertexModelSize * sizeof(GLfloat), this->nSpheres);

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glDisableVertexAttribArray(2);
        glDisableVertexAttribArray(3);
        glDisableVertexAttribArray(4);
        if (this->velocitiesX != nullptr) {
            glDisableVertexAttribArray(5);
            glDisableVertexAttribArray(6);
            glDisableVertexAttribArray(7);
        }

        // Swap front and back buffers
        glfwSwapBuffers(this->window);

        // Poll for and process events
        glfwPollEvents();

        // Sleep if necessary
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

// ==================================================================================== explicit template instantiation
template class OGLSpheresVisuInst<double>;
template class OGLSpheresVisuInst<float>;
// ==================================================================================== explicit template instantiation
#endif
