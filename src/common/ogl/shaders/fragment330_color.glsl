#version 330 core

// Interpolated values from the vertex shaders
in vec3 fColor;

// Ouput data
out vec3 outColor;

void main()
{
    // Output color = color specified in the vertex shader
    // interpolated between all 3 surrounding vertices
    outColor = fColor;
}
