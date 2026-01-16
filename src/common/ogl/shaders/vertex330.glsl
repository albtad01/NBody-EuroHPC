#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in float positionXPerVertex;
layout(location = 1) in float positionYPerVertex;
layout(location = 2) in float positionZPerVertex;

// Input colors
layout(location = 3) in float radiusPerVertex;

// Input colors
//layout(location = 4) in vec3 colorPerVertex;

// Output data ; will be interpolated for each fragment.
//out vec3 gColor;
out float gRadius;

void main()
{
    // Output position of the vertex, in clip space : MVP * position
    //gl_Position = MVP * vec4(vertexPosition, 1);
    gl_Position = vec4(positionXPerVertex, positionYPerVertex, positionZPerVertex, 1);

    // The color of each vertex will be interpolated
    // to produce the color of each fragment
    //gColor = colorPerVertex;

    gRadius = radiusPerVertex;
}
