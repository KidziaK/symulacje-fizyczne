import ctypes

import pygame as pg
import numpy as np

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader, ShaderProgram
from pathlib import Path
from camera import Camera
from shape import Cube, Shape
from enums import MouseButtons
from scipy.spatial.transform import Rotation


def gl_objects_from_shape(shape: Shape):
    vertex_stride = shape.vertices.shape[1] * 4

    vertex_array_object = glGenVertexArrays(1)
    glBindVertexArray(vertex_array_object)
    
    vertex_buffer_object = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object)
    glBufferData(GL_ARRAY_BUFFER, shape.vertices.nbytes, shape.vertices.flatten(), GL_STATIC_DRAW)

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, shape.vertices.shape[1] - 3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(12))


    element_buffer_object = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_object)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, shape.indices.nbytes, shape.indices.flatten(), GL_STATIC_DRAW)

    return vertex_array_object, vertex_buffer_object, element_buffer_object

class App:
    def __init__(self):
        pg.init()
        pg.display.set_mode((640, 480), pg.OPENGL|pg.DOUBLEBUF|pg.RESIZABLE)
        self.clock = pg.time.Clock()

        

        self.camera = Camera()

        glClearColor(0.3, 0.3, 0.3, 1)
        glEnable(GL_DEPTH_TEST)

        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) 

        self.program = self.create_program(Path("shaders/vertex.glsl"), Path("shaders/fragment.glsl"))
        glUseProgram(self.program)

        self.model_loc = glGetUniformLocation(self.program, "model")
        self.view_loc = glGetUniformLocation(self.program, "view")
        self.projection_loc = glGetUniformLocation(self.program, "projection")

        self.model_matrix = np.eye(4, dtype=np.float32)
        cube_rotation = Rotation.from_euler("xy", (45, 0), degrees=True)
        self.cube = Cube(rotation=cube_rotation.as_matrix().astype(np.float32))
        self.vao, self.vbo, self.ebo = gl_objects_from_shape(self.cube)
        
        self.set_projection_matrix()
        self.mainloop()

    def create_program(self, vertex_shader_path: Path, fragment_shader_path: Path) -> ShaderProgram:
        vertex_shader_txt = vertex_shader_path.read_text()
        fragment_shader_txt = fragment_shader_path.read_text()

        program = compileProgram(
            compileShader(vertex_shader_txt, GL_VERTEX_SHADER),
            compileShader(fragment_shader_txt, GL_FRAGMENT_SHADER)
        )

        return program
    
    def set_projection_matrix(self):
        width, height = pg.display.get_surface().get_size()
        
        glViewport(0, 0, width, height)
        
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, self.model_matrix)
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, self.camera.view_matrix)
        glUniformMatrix4fv(self.projection_loc, 1, GL_FALSE, self.camera.get_projection_matrix(width, height))

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.cube.index_count, GL_UNSIGNED_INT, None)

        glBindVertexArray(0)

    def mainloop(self):
        running = True
        is_dragging = False

        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False 
                elif event.type == pg.VIDEORESIZE:
                    self.set_projection_matrix() 
                elif event.type == pg.MOUSEBUTTONDOWN:
                    if event.button == MouseButtons.LEFT:  # Left click
                        is_dragging = True
                        last_mouse_pos = pg.mouse.get_pos()
                    elif event.button == MouseButtons.WHEEL_UP:
                        self.camera.zoom(1)
                        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, self.camera.view_matrix)
                    elif event.button == MouseButtons.WHEEL_DOWN:
                        self.camera.zoom(-1)
                        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, self.camera.view_matrix)
                    
                elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
                    is_dragging = False
                    last_mouse_pos = None
                    
                elif event.type == pg.MOUSEMOTION and is_dragging:
                    current_pos = pg.mouse.get_pos()
                    delta_x = current_pos[0] - last_mouse_pos[0]
                    delta_y = last_mouse_pos[1] - current_pos[1]
                    self.camera.rotate(delta_x, delta_y)
                    last_mouse_pos = current_pos
                    glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, self.camera.view_matrix)


            self.render()
            pg.display.flip()
            self.clock.tick(60)

        self.quit()

    def quit(self):
        pg.quit()


if __name__ == "__main__":
    app = App()
    