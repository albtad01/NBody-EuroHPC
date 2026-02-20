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

        // 1st attribute buffer : vertex positions
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

        // 3rd attribute buffer : vertex velocities / colors
        if (this->velocitiesX && this->color) {
            
            // --- CYBERPUNK AUDIO-SYNC LOGIC ---
            
            // 1. Get current time for synchronization
            double time = glfwGetTime();
            
            // "Move Your Body" by Eiffel 65 is approx 130 BPM.
            // 130 beats / 60 seconds = 2.166 Hz.
            double bpm = 130.0;
            double freq = bpm / 60.0;
            
            // beatPhase is the current angle in the rhythmic cycle
            double beatPhase = time * freq * 2.0 * 3.14159;

            // Create a sharp "Kick" impulse (0.0 to 1.0)
            // Power of 8 compresses the wave to simulate a drum kick (short peak, long silence)
            float beatPulse = (float)pow((sin(beatPhase) + 1.0) / 2.0, 8.0); 

            // compute colors: First pass to find Min/Max velocity for normalization
            float min = std::numeric_limits<float>::max();
            float max = std::numeric_limits<float>::min();
            
            for (long unsigned int i = 0; i < this->nSpheres; i++) {
                const float accX = this->velocitiesXBuffer[i];
                const float accY = this->velocitiesYBuffer[i];
                const float accZ = this->velocitiesZBuffer[i];
                
                // Calculate squared norm (approximation of speed/energy)
                const float norm = accX * accX + accY * accY + accZ * accZ;
                
                // Temporarily store the norm in the Red channel to reuse it in the next loop
                this->colorBuffer[i * 3 + 0] = norm; 
                
                min = std::min(min, norm);
                max = std::max(max, norm);
            }

            // Second pass: Apply Cyberpunk Colors
            for (long unsigned int i = 0; i < this->nSpheres; i++) {
                const float norm = this->colorBuffer[i * 3 + 0];
                
                // 't' is the normalized velocity factor: 0.0 (Slowest) -> 1.0 (Fastest)
                float t = (norm - min) / (max - min + 1e-6f); // +epsilon to avoid div by zero
                
                // --- DYNAMIC CYBERPUNK PALETTE ---
                
                // 1. Slow Bodies (t close to 0): Deep Space Blue/Black
                // They blend with the background to emphasize the structure
                float r = 0.0f;
                float g = 0.02f;
                float b = 0.1f;

                // 2. Fast Bodies (t increases): Gradient to Electric Cyan
                // We mix in the "Speed Color" based on 't'
                if (t > 0.1f) {
                    // Linear interpolation towards Cyan (0, 1, 1)
                    r += t * 0.1f;  // Slight white tint for very fast ones
                    g += t * 0.9f;  // Green ramps up to create Cyan
                    b += t * 1.5f;  // Blue saturates quickly
                }

                // 3. STROBE EFFECT (Bass/Kick)
                // When the music hits ('beatPulse' is high), we flash the fast particles
                // This creates the "Energy Core" pulsing effect
                if (t > 0.25f) {
                    float flash = beatPulse * 0.8f; // Flash intensity
                    
                    // Add flash to all channels (Whitening)
                    r += flash;
                    g += flash;
                    b += flash;
                }

                // 4. Hyper-Speed Glow
                // Very fast particles are always bright white/cyan
                if (t > 0.8f) {
                    r = 0.8f + beatPulse * 0.2f;
                    g = 1.0f;
                    b = 1.0f;
                }

                // Final Clamp to ensure valid color range [0.0, 1.0]
                this->colorBuffer[i * 3 + 0] = std::min(r, 1.0f);
                this->colorBuffer[i * 3 + 1] = std::min(g, 1.0f);
                this->colorBuffer[i * 3 + 2] = std::min(b, 1.0f);
            }

            glEnableVertexAttribArray(iBufferIndex);
            glBindBuffer(GL_ARRAY_BUFFER, this->colorBufferRef);
            glVertexAttribPointer(
                iBufferIndex++, // attribute.
                3,              // size (R, G, B)
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

        // Draw the particles!
        glDrawArrays(GL_POINTS, 0, this->nSpheres);

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glDisableVertexAttribArray(2);
        glDisableVertexAttribArray(3);

        // Swap front and back buffers
        glfwSwapBuffers(this->window);

        // Poll for and process events
        glfwPollEvents();
    }
}

// ==================================================================================== explicit template instantiation
template class OGLSpheresVisuGS<double>;
template class OGLSpheresVisuGS<float>;
// ==================================================================================== explicit template instantiation
#endif