#ifdef VISU
#include "OGLControl.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <thread>

OGLControl::OGLControl(GLFWwindow *window)
    : window(window), camPosition(glm::vec3(0, 0, 5)), direction(glm::vec3(0, 0, -1)), right(glm::vec3(1, 0, 0)),
      up(glm::vec3(0, 1, 0)), horizontalAngle(3.14f), // Initial horizontal angle : toward -Z
      verticalAngle(0.0f),                            // Initial vertical angle : none
      initialFoV(45.0f),                              // Initial Field of View
      speed(3.0f),                                    // 3 units per second
      mouseSpeed(0.005f), xMousePos(-1), yMousePos(-1), lastTime(-1)
{
    int winWidth, winHeight;
    glfwGetWindowSize(this->window, &winWidth, &winHeight);
    this->projectionMatrix = glm::perspective(this->initialFoV, (winWidth * 1.0f) / (winHeight * 1.0f), 0.1f, 5000.0f);
}

OGLControl::~OGLControl() {}

glm::mat4 OGLControl::computeViewAndProjectionMatricesFromInputs()
{
    // glfwGetTime is called only once, the first time this function is called
    if (this->lastTime == -1)
        this->lastTime = glfwGetTime();

    // Compute time difference between current and last frame
    double currentTime = glfwGetTime();
    float deltaTime = float(currentTime - lastTime);

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_RELEASE) {
        this->xMousePos = -1;
        this->yMousePos = -1;
    }

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS) {
        double curXMousePos, curYMousePos;
        glfwGetCursorPos(window, &curXMousePos, &curYMousePos);

        if (this->xMousePos == -1 && this->yMousePos == -1)
            glfwGetCursorPos(window, &this->xMousePos, &this->yMousePos);
        else
            // Reset mouse position for next frame
            glfwSetCursorPos(window, this->xMousePos, this->yMousePos);

        // std::cout << "GLFW_MOUSE_BUTTON_1 == GLFW_PRESS" << std::endl;

        // Compute new orientation
        this->horizontalAngle += mouseSpeed * float(this->xMousePos - curXMousePos);
        this->verticalAngle += mouseSpeed * float(this->yMousePos - curYMousePos);

        // Direction : Spherical coordinates to Cartesian coordinates conversion
        this->direction.x = cos(this->verticalAngle) * sin(this->horizontalAngle);
        this->direction.y = sin(this->verticalAngle);
        this->direction.z = cos(this->verticalAngle) * cos(this->horizontalAngle);

        // Right vector
        this->right.x = sin(this->horizontalAngle - 3.14f / 2.0f);
        this->right.y = 0;
        this->right.z = cos(this->horizontalAngle - 3.14f / 2.0f);

        // Up vector
        up = glm::cross(right, direction);
    }

    // Move forward
    if ((glfwGetKey(window, GLFW_KEY_UP) || glfwGetKey(window, GLFW_KEY_W)) == GLFW_PRESS) {
        this->camPosition += this->direction * deltaTime * this->speed;
    }
    // Move backward
    if ((glfwGetKey(window, GLFW_KEY_DOWN) || glfwGetKey(window, GLFW_KEY_S)) == GLFW_PRESS) {
        this->camPosition -= this->direction * deltaTime * this->speed;
    }
    // Strafe right
    if ((glfwGetKey(window, GLFW_KEY_RIGHT) || glfwGetKey(window, GLFW_KEY_D)) == GLFW_PRESS) {
        this->camPosition += this->right * deltaTime * this->speed;
    }
    // Strafe left
    if ((glfwGetKey(window, GLFW_KEY_LEFT) || glfwGetKey(window, GLFW_KEY_A)) == GLFW_PRESS) {
        this->camPosition -= this->right * deltaTime * this->speed;
    }

    /*
    std::cout << "this->camPosition.x = " << this->camPosition.x << ", "
              << "this->camPosition.y = " << this->camPosition.y << ", "
              << "this->camPosition.z = " << this->camPosition.z << std::endl;

    std::cout << "this->direction.x = " << this->direction.x << ", "
              << "this->direction.y = " << this->direction.y << ", "
              << "this->direction.z = " << this->direction.z << std::endl;

    std::cout << "this->camPosition.x + this->direction.x = " << this->camPosition.x + this->direction.x << ", "
              << "this->camPosition.y + this->direction.y = " << this->camPosition.y + this->direction.y << ", "
              << "this->camPosition.z + this->direction.z = " << this->camPosition.z + this->direction.z << std::endl;
    */

    // Camera matrix
    this->viewMatrix =
        glm::lookAt(this->camPosition,                   // Camera is here
                    this->camPosition + this->direction, // and looks here : at the same position, plus "direction"
                    this->up);                           // head is up (set to 0,-1,0 to look upside-down)

    // For the next frame, the "last time" will be "now"
    this->lastTime = currentTime;

    return this->projectionMatrix * this->viewMatrix;
}

glm::mat4 OGLControl::getViewMatrix() { return this->viewMatrix; }

glm::mat4 OGLControl::getProjectionMatrix() { return this->projectionMatrix; }
#endif /* VISU */
