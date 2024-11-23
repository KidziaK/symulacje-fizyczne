import ctypes

import pygame as pg
import OpenGL.GL as gl
import numpy as np

from OpenGL.GL.shaders import compileProgram, compileShader, ShaderProgram
from pathlib import Path

class App:
    def __init__(self):
        pg.init()
        pg.display.set_mode((640, 480), pg.OPENGL|pg.DOUBLEBUF)
        self.clock = pg.time.Clock()

        gl.glClearColor(0.3, 0.3, 0.3, 1)

        self.program = self.create_program(Path("shaders/vertex.glsl"), Path("shaders/fragment.glsl"))
        gl.glUseProgram(self.program)

        self.triangle = Triangle()
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        self.mainloop()

    def create_program(self, vertex_shader_path: Path, fragment_shader_path: Path) -> ShaderProgram:
        vertex_shader_txt = vertex_shader_path.read_text()
        fragment_shader_txt = fragment_shader_path.read_text()

        program = compileProgram(
            compileShader(vertex_shader_txt, gl.GL_VERTEX_SHADER),
            compileShader(fragment_shader_txt, gl.GL_FRAGMENT_SHADER)
        )

        return program

    def mainloop(self):
        running = True
        while(running):
            for event in pg.event.get():
                if (event.type == pg.QUIT):
                    running = False 
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            gl.glUseProgram(self.program)
            gl.glBindVertexArray(self.triangle.vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.triangle.vertex_count)


            pg.display.flip()

            self.clock.tick(60)
        self.quit()

    def quit(self):
        pg.quit()

class Triangle:
    def __init__(self) -> None:
        self.vertices = (
            -0.5, -0.5, 0.0, 1.0, 0.0, 0.0,
             0.5, -0.5, 0.0, 0.0, 1.0, 0.0,
             0.0,  0.5, 0.0, 0.0, 0.0, 1.0
        )

        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vertex_count = 3

        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, gl.GL_STATIC_DRAW)

        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(12))

if __name__ == "__main__":
    app = App()