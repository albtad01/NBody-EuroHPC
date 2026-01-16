#ifdef VISU
#ifndef OGL_CONTROL_HPP_
#define OGL_CONTROL_HPP_

#include <GLFW/glfw3.h>

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

class OGLControl {
  private:
    GLFWwindow *window;

    glm::vec3 camPosition; // = glm::vec3( 0, 0, 5 );

    glm::vec3 direction; // = glm::vec3( 0, 0, -1 );
    glm::vec3 right;     // = glm::vec3( 1, 0, 0 );
    glm::vec3 up;        // = glm::vec3( 0, 1, 0 );

    glm::mat4 viewMatrix;
    glm::mat4 projectionMatrix;

    float horizontalAngle; // = 3.14f - Initial horizontal angle : toward -Z
    float verticalAngle;   // = 0.0f  - Initial vertical angle : none
    float initialFoV;      // = 45.0f - Initial Field of View

    float speed;      // = 3.0f - 3 units per second
    float mouseSpeed; // = 0.005f

    // Get mouse position
    double xMousePos; // = -1
    double yMousePos; // = -1

    // Keep last time
    double lastTime; // -1

  public:
    OGLControl(GLFWwindow *window);

    virtual ~OGLControl();

    glm::mat4 computeViewAndProjectionMatricesFromInputs();

    glm::mat4 getViewMatrix();

    glm::mat4 getProjectionMatrix();
};

#endif /* OGL_CONTROL_HPP_ */
#endif /* VISU */
