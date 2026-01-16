#ifdef VISU
#include <cstdlib>
#include <fstream>
#include <iostream>
using namespace std;

#include <GL/glew.h>

#include "OGLTools.hpp"

GLFWwindow *OGLTools::initAndMakeWindow(const int winWidth, const int winHeight, const string winName)
{
    GLFWwindow *window;

    /* Initialize the library */
    if (!glfwInit()) {
        cerr << "Failed to initialize GLFW." << endl;
        return (GLFWwindow *)0;
    }

    /* window conf */
    glfwWindowHint(GLFW_SAMPLES, 8); // anti-aliasing x8
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, 0); // user can't resize window

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(winWidth, winHeight, winName.c_str(), NULL, NULL);

    if (!window) {
        cerr << "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible." << endl;
        glfwTerminate();
        return (GLFWwindow *)0;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    glewExperimental = true; // Needed for core profile
    if (glewInit() != GLEW_OK) {
        cerr << "Failed to initialize GLEW." << endl;
        return (GLFWwindow *)0;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    return window;
}

GLuint OGLTools::loadShaderFromFile(const GLenum shaderType, const string shaderFilePath)
{
    // create the shader
    GLuint shader = glCreateShader(shaderType);

    // Read the shader code from the file
    string shaderCode;
    ifstream shaderStream(shaderFilePath.c_str(), ios::in);
    if (shaderStream.is_open()) {
        string line = "";
        while (getline(shaderStream, line))
            shaderCode += "\n" + line;
        shaderStream.close();
    }
    else {
        cout << "Impossible to open " << shaderFilePath << ". Are you in the right directory ?" << endl;
        return (GLuint)0;
    }

    // compile shader
    cout << "Compiling shader: " << shaderFilePath << endl;
    const char *sourcePointer = shaderCode.c_str();
    glShaderSource(shader, 1, &sourcePointer, NULL);
    glCompileShader(shader);

    // check the shader
    GLint result = GL_FALSE;
    int infoLogLength;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &result);
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);
    if (infoLogLength > 1) {
        vector<char> shaderErrorMessage(infoLogLength + 1);
        glGetShaderInfoLog(shader, infoLogLength, NULL, &shaderErrorMessage[0]);
        cout << &shaderErrorMessage[0] << endl;
        return ((GLuint)0);
    }

    return shader;
}

GLuint OGLTools::linkShaders(const vector<GLuint> shaders)
{
    GLuint shaderProgram = glCreateProgram();

    cout << "Linking shader program...";

    for (unsigned i = 0; i < shaders.size(); i++)
        glAttachShader(shaderProgram, shaders[i]);
    glLinkProgram(shaderProgram);

    /* check the shader program */
    GLint result = GL_FALSE;
    int infoLogLength;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &result);
    glGetProgramiv(shaderProgram, GL_INFO_LOG_LENGTH, &infoLogLength);
    if (infoLogLength > 1) {
        cout << " FAILED !" << endl;
        vector<char> programErrorMessage(infoLogLength + 1);
        glGetProgramInfoLog(shaderProgram, infoLogLength, NULL, &programErrorMessage[0]);
        cout << &programErrorMessage[0] << endl;
        return ((GLuint)0);
    }
    else
        cout << " SUCCESS !" << endl;

    return (shaderProgram);
}
#endif /* VISU */
