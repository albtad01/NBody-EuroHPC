#version 330 core

#define N_POINTS_PER_CIRCLE 22 // compulsory even number !

const float PI = 3.1415926;

layout(points) in;
//layout(line_strip, max_vertices = 256) out; // no color
layout(line_strip, max_vertices = 146) out; // color

in vec3  gColor []; // Output from vertex shader for each vertex
in float gRadius[];

out vec3 fColor; // Output to fragment shader

// Values that stay constant for the whole mesh.
uniform mat4 MVP;

void main() {
    fColor = gColor[0]; // Point has only one vertex

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

    /*
    // draw a plain sphere
    for (int j = 0; j <= N_POINTS_PER_CIRCLE; j++) {
        float horizontalAngle = (PI * 2.0 / N_POINTS_PER_CIRCLE) * j;

        for (int i = 0; i <= N_POINTS_PER_CIRCLE; i += 2) {
            // Angle between each side in radiant
            float verticalAngle = (PI * 2.0 / N_POINTS_PER_CIRCLE) * i;

            vec4 offset;
            // Offset from center of point
            offset = scale * vec4(cos(verticalAngle) * sin(horizontalAngle),
                                  sin(verticalAngle),
                                  cos(verticalAngle) * cos(horizontalAngle),
                                  0.0);
            gl_Position = MVP * (screenPos + offset);
            EmitVertex();

            verticalAngle = (PI * 2.0 / N_POINTS_PER_CIRCLE) * (i +1);
            offset = scale * vec4(cos(verticalAngle) * sin(horizontalAngle),
                                  sin(verticalAngle),
                                  cos(verticalAngle) * cos(horizontalAngle),
                                  0.0);
            gl_Position = MVP * (screenPos + offset);
            EmitVertex();

            horizontalAngle = (PI * 2.0 / N_POINTS_PER_CIRCLE) * (j +1);
            offset = scale * vec4(cos(verticalAngle) * sin(horizontalAngle),
                                  sin(verticalAngle),
                                  cos(verticalAngle) * cos(horizontalAngle),
                                  0.0);
            gl_Position = MVP * (screenPos + offset);
            EmitVertex();

            verticalAngle = (PI * 2.0 / N_POINTS_PER_CIRCLE) * (i +2);
            offset = scale * vec4(cos(verticalAngle) * sin(horizontalAngle),
                                  sin(verticalAngle),
                                  cos(verticalAngle) * cos(horizontalAngle),
                                  0.0);
            gl_Position = MVP * (screenPos + offset);
            EmitVertex();


            horizontalAngle = (PI * 2.0 / N_POINTS_PER_CIRCLE) * j;
            verticalAngle = (PI * 2.0 / N_POINTS_PER_CIRCLE) * (i +1);
            offset = scale * vec4(cos(verticalAngle) * sin(horizontalAngle),
                                  sin(verticalAngle),
                                  cos(verticalAngle) * cos(horizontalAngle),
                                  0.0);
            gl_Position = MVP * (screenPos + offset);
            EmitVertex();
        }
    }
    */

    EndPrimitive();
}
