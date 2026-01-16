#version 130

// Input color
in float colorRPerVertex;
in float colorGPerVertex;
in float colorBPerVertex;

// Ouput data
out vec3 outColor;

void main()
{
    vec3 color = vec3(colorRPerVertex, colorGPerVertex, colorBPerVertex);
    outColor = color;
}
