#version 330 core

in vec4 fragmentCol;

out vec4 color;

void main()
{
    color = fragmentCol;
}