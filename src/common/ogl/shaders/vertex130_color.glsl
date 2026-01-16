#version 130

// Input vertex data, different for all executions of this shader.
in vec3  modelPerVertex;
in float positionXPerVertex;
in float positionYPerVertex;
in float positionZPerVertex;

in float accXPerVertex;
in float accYPerVertex;
in float accZPerVertex;

// Input radius
in float radiusPerVertex;

// Output color
out float colorRPerVertex;
out float colorGPerVertex;
out float colorBPerVertex;

// Values that stay constant for the whole mesh.
uniform mat4 MVP;

void main()
{
    float scale = radiusPerVertex * 1.0e-8f;

     // trick in order to avoid start point at the center of the sphere
    if(modelPerVertex.x == 0.0f && modelPerVertex.y == 0.0f && modelPerVertex.z == 0.0f)
        gl_Position = MVP * vec4(1.0e-8f * positionXPerVertex + (scale * 0),
                                 1.0e-8f * positionYPerVertex + (scale * 1),
                                 1.0e-8f * positionZPerVertex + (scale * 0),
                                 1);
    else
        gl_Position = MVP * vec4(1.0e-8f * positionXPerVertex + (scale * modelPerVertex.x),
                                 1.0e-8f * positionYPerVertex + (scale * modelPerVertex.y),
                                 1.0e-8f * positionZPerVertex + (scale * modelPerVertex.z),
                                 1);

    float norm = accXPerVertex * accXPerVertex + accYPerVertex * accYPerVertex + accZPerVertex * accZPerVertex;

    float normX = accXPerVertex * accXPerVertex;
    float normY = accYPerVertex * accYPerVertex;
    float normZ = accZPerVertex * accZPerVertex;

    // colorRPerVertex = normX * normY * normZ * 1.75e5f;
    // colorGPerVertex = radiusPerVertex * 9.0e-7f;
    // colorBPerVertex = norm;

    colorRPerVertex = normX * 1.75e5f;
    colorGPerVertex = normY * 1.75e5f;
    colorBPerVertex = normZ * 1.75e5f;
}
