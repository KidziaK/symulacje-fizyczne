#version 330 core

layout (location = 0) in vec3 vertexPos;
layout (location = 1) in vec4 vertexCol;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec4 fragmentCol;

void main() 
{
    gl_Position = projection * view * model * vec4(vertexPos, 1.0);
    fragmentCol = vertexCol;
}