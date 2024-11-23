#version 330 core

in vec3 fragmentCol;

out vec4 color;

void main()
{
    color = vec4(fragmentCol, 1.0f);
}