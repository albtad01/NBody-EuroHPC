#version 330 core

// Interpolated values from the vertex shaders
in vec3 fColor;

// Ouput data
out vec3 outColor;

void main()
{
    // vec3 white      = vec3(1, 1, 1);
    // //vec3 darkBrown  = vec3(0.43f, 0.22f, 0.0f);
    // //vec3 lightBrown = vec3(0.75f, 0.45f, 0.10f);
    // outColor = white;

    // Output color = color specified in the vertex shader
    // interpolated between all 3 surrounding vertices
    outColor = fColor;
}
