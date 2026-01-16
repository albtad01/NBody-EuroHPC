#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in float positionXPerVertex;
layout(location = 1) in float positionYPerVertex;
layout(location = 2) in float positionZPerVertex;

// Input colors
layout(location = 3) in float radiusPerVertex;

// Input acc data
layout(location = 4) in float accXPerVertex;
layout(location = 5) in float accYPerVertex;
layout(location = 6) in float accZPerVertex;

// Output data ; will be interpolated for each fragment.
out float gRadius;

// Output color
out vec3 gColor;

float MAPPING_R[16] = float[](66, 25,  9,  4,   0,  12,  24,  57, 134, 211, 241, 248, 255, 204, 153, 106);
float MAPPING_G[16] = float[](30,  7,  1,  4,   7,  44,  82, 125, 181, 236, 233, 201, 170, 128,  87,  52);
float MAPPING_B[16] = float[](15, 26, 47, 73, 100, 138, 177, 209, 229, 248, 191,  95,   0,   0,   0,   3);

void main()
{
    // Output position of the vertex, in clip space : MVP * position
    //gl_Position = MVP * vec4(vertexPosition, 1);
    gl_Position = vec4(positionXPerVertex, positionYPerVertex, positionZPerVertex, 1);

    gRadius = radiusPerVertex;

    float normX = accXPerVertex * accXPerVertex;
    float normY = accYPerVertex * accYPerVertex;
    float normZ = accZPerVertex * accZPerVertex;

    float norm = normX + normY + normZ;
    norm = norm * 2000000;
    int n = int(norm);
    float red   = MAPPING_R[n%16] / 255;
    float green = MAPPING_G[n%16] / 255;
    float blue  = MAPPING_B[n%16] / 255;

    gColor = vec3(red, green, blue);
}
