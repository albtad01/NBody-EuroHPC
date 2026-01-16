#ifdef VISU
#ifndef OGLTOOLS_HPP_
#define OGLTOOLS_HPP_

#include <string>
#include <vector>

#include <GLFW/glfw3.h>

class OGLTools {
  public:
    static GLFWwindow *initAndMakeWindow(const int winWidth, const int winHeight, const std::string winName);

    static GLuint loadShaderFromFile(const GLenum shaderType, const std::string shaderFilePath);

    static GLuint linkShaders(const std::vector<GLuint> shaders);
};

#endif /* OGLTOOLS_HPP_ */
#endif /* VISU */
