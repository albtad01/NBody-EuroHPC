#version 150
#extension GL_EXT_gpu_shader4: enable
#extension GL_EXT_geometry_shader4: enable

#define N_POINTS_PER_CIRCLE 4 // compulsory even number !

const float PI = 3.1415926;

layout(points) in;
layout(line_strip, max_vertices = 256) out; // no color

in float gRadius[];

// Values that stay constant for the whole mesh.
uniform mat4 MVP;

void main() {
    vec4 scale = vec4(gRadius[0] * 1.0e-8f);

    vec4 screenPos = vec4(gl_in[0].gl_Position.x * 1.0e-8f,
                          gl_in[0].gl_Position.y * 1.0e-8f,
                          gl_in[0].gl_Position.z * 1.0e-8f,
                          1);

    // draw a empty sphere
    for (int j = 0; j <= (N_POINTS_PER_CIRCLE / 2); j++) {
        float horizontalAngle = (PI * 2.0 / N_POINTS_PER_CIRCLE) * j;

        for (int i = 0; i <= N_POINTS_PER_CIRCLE; i++) {
            // Angle between each side in radiant
            float verticalAngle = (PI * 2.0 / N_POINTS_PER_CIRCLE) * i + (3.14f / 2.0f);

            // Offset from center of point
            vec4 offset = scale * vec4(cos(verticalAngle) * sin(horizontalAngle),
                                       sin(verticalAngle),
                                       cos(verticalAngle) * cos(horizontalAngle),
                                       0.0);
            gl_Position = MVP * (screenPos + offset);
            EmitVertex();
        }
    }

    EndPrimitive();
}
