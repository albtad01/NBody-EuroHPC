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

#include "OGLSpheresVisuGS.hpp"
#include "OGLTools.hpp"

template <typename T>
OGLSpheresVisuGS<T>::OGLSpheresVisuGS(const std::string winName, const int winWidth, const int winHeight,
                                      const T *positionsX, const T *positionsY, const T *positionsZ,
                                      const T *velocitiesX, const T *velocitiesY, const T *velocitiesZ, const T *radius,
                                      const unsigned long nSpheres, const bool color)
    : OGLSpheresVisu<T>(winName, winWidth, winHeight, positionsX, positionsY, positionsZ, velocitiesX,
                        velocitiesX ? velocitiesY : nullptr, velocitiesX ? velocitiesZ : nullptr, radius, nSpheres,
                        color)
{
    if (this->window) {
        // specify shaders path and compile them
        std::vector<GLenum> shadersType(3);
        std::vector<std::string> shadersFiles(3);
        shadersType[0] = GL_VERTEX_SHADER;
        shadersFiles[0] = velocitiesX && color ? "../src/common/ogl/shaders/vertex330_color_v2.glsl"
                                               : "../src/common/ogl/shaders/vertex330.glsl";
        shadersType[1] = GL_GEOMETRY_SHADER;
        shadersFiles[1] = velocitiesX && color ? "../src/common/ogl/shaders/geometry330_color_v2.glsl"
                                               : "../src/common/ogl/shaders/geometry330.glsl";
        shadersType[2] = GL_FRAGMENT_SHADER;
        shadersFiles[2] = velocitiesX && color ? "../src/common/ogl/shaders/fragment330_color_v2.glsl"
                                               : "../src/common/ogl/shaders/fragment330.glsl";

        this->compileShaders(shadersType, shadersFiles);
    }
}

template <typename T> OGLSpheresVisuGS<T>::~OGLSpheresVisuGS() {}

template <typename T> void OGLSpheresVisuGS<T>::refreshDisplay()
{
    if (this->window) {
        this->updatePositions();

        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // use our shader program
        if (this->shaderProgramRef != 0)
            glUseProgram(this->shaderProgramRef);

        // 1rst attribute buffer : vertex positions
        int iBufferIndex;
        for (iBufferIndex = 0; iBufferIndex < 3; iBufferIndex++) {
            glEnableVertexAttribArray(iBufferIndex);
            glBindBuffer(GL_ARRAY_BUFFER, this->positionBufferRef[iBufferIndex]);
            glVertexAttribPointer(
                iBufferIndex, // attribute. No particular reason for 0, but must match the layout in the shader.
                1,            // size
                GL_FLOAT,     // type
                GL_FALSE,     // normalized?
                0,            // stride
                (void *)0     // array buffer offset
            );
        }

        // 2nd attribute buffer : radius
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

        // 3rd attribute buffer : vertex velocities
        if (this->velocitiesX && this->color) {
            // for (int i = 0; i < 3; i++) {
            //     glEnableVertexAttribArray(iBufferIndex);
            //     glBindBuffer(GL_ARRAY_BUFFER, this->accelerationBufferRef[i]);
            //     glVertexAttribPointer(
            //         iBufferIndex++, // attribute. No particular reason for 0, but must match the layout in the
            //         shader. 1,              // size GL_FLOAT,       // type GL_FALSE,       // normalized? 0, //
            //         stride (void *)0       // array buffer offset
            //     );
            // }

            // compute colors
            float min = std::numeric_limits<float>::max();
            float max = std::numeric_limits<float>::min();
            for (long unsigned int i = 0; i < this->nSpheres; i++) {
                const float accXPerVertex = this->velocitiesXBuffer[i];
                const float accYPerVertex = this->velocitiesYBuffer[i];
                const float accZPerVertex = this->velocitiesZBuffer[i];

                const float normX = accXPerVertex * accXPerVertex;
                const float normY = accYPerVertex * accYPerVertex;
                const float normZ = accZPerVertex * accZPerVertex;

                const float norm = normX + normY + normZ;

                this->colorBuffer[i * 3 + 0] = norm;

                min = std::min(min, norm);
                max = std::max(max, norm);
            }

            static uint8_t MAPPING_R[16] = {106, 153, 204, 255, 248, 241, 211, 134,  57,  24,  12,   0,  4,  9, 25, 66};
            static uint8_t MAPPING_G[16] = { 52,  87, 128, 170, 201, 233, 236, 181, 125,  82,  44,   7,  4,  1,  7, 30};
            static uint8_t MAPPING_B[16] = {  3,   0,   0,   0,  95, 191, 248, 229, 209, 177, 138, 100, 73, 47, 26, 15};

            // static uint8_t MAPPING_R[10] = { 3, 55, 106, 157, 208, 220, 232, 244, 250, 255};
            // static uint8_t MAPPING_G[10] = { 7,  6,   4,   2,   0,  47,  93, 140, 163, 186};
            // static uint8_t MAPPING_B[10] = {30, 23,  15,   8,   0,   2,   4,   6,   7,   8};

            for (long unsigned int i = 0; i < this->nSpheres; i++) {
                const float norm = this->colorBuffer[i * 3 + 0];

                // const unsigned colorRange = 2 * sizeof(MAPPING_R) -1;
                const unsigned colorRange = 1 * sizeof(MAPPING_R) - 1;

                const float mix = (norm - min) / (max - min);
                const int n = (int)(mix * colorRange);

                const float red = MAPPING_R[(n) % sizeof(MAPPING_R)] / 255.f;
                const float green = MAPPING_G[(n) % sizeof(MAPPING_G)] / 255.f;
                const float blue = MAPPING_B[(n) % sizeof(MAPPING_B)] / 255.f;

                this->colorBuffer[i * 3 + 0] = red;
                this->colorBuffer[i * 3 + 1] = green;
                this->colorBuffer[i * 3 + 2] = blue;
            }

            glEnableVertexAttribArray(iBufferIndex);
            glBindBuffer(GL_ARRAY_BUFFER, this->colorBufferRef);
            glVertexAttribPointer(
                iBufferIndex++, // attribute. No particular reason for 0, but must match the layout in the shader.
                3,              // size
                GL_FLOAT,       // type
                GL_FALSE,       // normalized?
                0,              // stride
                (void *)0       // array buffer offset
            );
        }

        // Compute the MVP matrix from keyboard and mouse input
        this->mvp = this->control->computeViewAndProjectionMatricesFromInputs();

        // Send our transformation to the currently bound shader,
        // in the "MVP" uniform
        glUniformMatrix4fv(this->mvpRef, 1, GL_FALSE, &this->mvp[0][0]);

        // Draw the triangle !
        glDrawArrays(GL_POINTS, 0, this->nSpheres);

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glDisableVertexAttribArray(2);
        glDisableVertexAttribArray(3);

        // Swap front and back buffers
        glfwSwapBuffers(this->window);

        // Poll for and process events
        glfwPollEvents();

        // Sleep if necessary
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

// ==================================================================================== explicit template instantiation
template class OGLSpheresVisuGS<double>;
template class OGLSpheresVisuGS<float>;
// ==================================================================================== explicit template instantiation
#endif
