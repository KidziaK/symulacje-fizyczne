import itertools

import imgui
import pygame

import numpy as np

from dataclasses import dataclass, field
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from imgui.integrations.pygame import PygameRenderer
from pygame.locals import *
from scipy.spatial.transform import Rotation


@dataclass
class SimulationParameters:
    m: float = 1.0
    c1: float = 36
    k: float = 4
    c2: float = 33
    a: float = 3
    mu: float = 0.5
    show_control_points: bool = True
    show_springs: bool = True
    show_control_frame: bool = True
    show_axes: bool = True
    show_faces: bool = True
    h: float = 0.01
    translation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    rotation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))


class BezierCubeSimulation:
    def __init__(self):
        glutInit(sys.argv)

        pygame.init()
        self.display = (1500, 900)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)

        imgui.create_context()
        self.imgui_renderer = PygameRenderer()
        self.controls_size_px = 400

        self.io = imgui.get_io()
        self.io.display_size = self.display

        glClearColor(0.3, 0.3, 0.3, 1.0)

        self._setup_camera()

        glEnable(GL_DEPTH_TEST)

        self.params = SimulationParameters()

        self.camera_rotation = np.array([0, 0, 0])
        self.camera_position = np.array([0, 0, -10])
        self.scale = 1.0

        self.initial_points, self.control_points = self._initialize_control_points()

        self.initial_frame = self._initialize_control_frame()
        self.control_frame = self._initialize_control_frame()

        self.bounding_box = [(-5, -5, -5), (5, 5, 5)]

        self.X = self.control_points.reshape(64, 3)
        self.V = np.zeros(shape=(64, 3))
        self.L = np.zeros(shape=(64, 64))

        self.neighbors = np.zeros(shape=(64, 64), dtype=bool)

        self.simulating = True

        for i in range(64):
            v1 = self.initial_points[i]

            for j in range(64):
                v2 = self.initial_points[j]

                distance = np.linalg.norm(v1 - v2)
                self.L[i, j] = distance

                if 0 < distance < 1.5:
                    self.neighbors[i, j] = True


        self.idx_cube = np.array([0, 3, 12, 15, 48, 51, 60, 63])
        self.idx_frame = np.array([0, 4, 3, 7, 1, 5, 2, 6])

    def _setup_camera(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -10.0)

    def _initialize_control_points(self):
        points = np.zeros(shape=(4, 4, 4, 3), dtype=np.float32)
        for x, y, z in itertools.product(range(4), range(4), range(4)):
            px = x - self.params.a / 2
            py = y - self.params.a / 2
            pz = z - self.params.a / 2
            points[x, y, z] = px, py, pz
        return points.reshape(64, 3), points.copy() + np.random.uniform(-0.2, 0.2, size=(4, 4, 4, 3))

    def _initialize_control_frame(self):
        a = self.params.a / 2

        return np.array([
            [-a, -a, -a],
            [a, -a, -a],
            [a, a, -a],
            [-a, a, -a],
            [-a, -a, a],
            [a, -a, a],
            [a, a, a],
            [-a, a, a],
        ])

    def draw_control_points(self):
        glPointSize(5)
        glBegin(GL_POINTS)
        glColor3f(1, 0, 0)
        np.apply_along_axis(glVertex3fv, axis=1, arr=self.X)

        glEnd()

        if self.params.show_springs:
            glBegin(GL_LINES)
            glColor3f(0, 0, 0)

            for i in range(64):
                for j in range(64):
                    if self.neighbors[i, j]:
                        glColor3f(0.7, 0.7, 0.7)
                        glVertex3fv(self.X[i])
                        glVertex3fv(self.X[j])

            glEnd()

    def draw_control_frame(self):
        glBegin(GL_LINES)
        glColor3f(0, 1, 0)
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        for edge in edges:
            glVertex3fv(self.control_frame[edge[0]])
            glVertex3fv(self.control_frame[edge[1]])
        glEnd()

    def draw_bounding_box(self):
        glBegin(GL_LINES)
        glColor3f(0, 0, 1)
        min_point, max_point = self.bounding_box

        vertices = [
            (min_point[0], min_point[1], min_point[2]),
            (max_point[0], min_point[1], min_point[2]),
            (max_point[0], max_point[1], min_point[2]),
            (min_point[0], max_point[1], min_point[2]),
            (min_point[0], min_point[1], max_point[2]),
            (max_point[0], min_point[1], max_point[2]),
            (max_point[0], max_point[1], max_point[2]),
            (min_point[0], max_point[1], max_point[2]),
        ]

        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]

        for edge in edges:
            glVertex3fv(vertices[edge[0]])
            glVertex3fv(vertices[edge[1]])

        glEnd()

    def draw_axes(self):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.display[0], 0, self.display[1], -100, 100)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)

        base_x = self.display[0] - 80
        base_y = 60
        arrow_length = 200
        arrow_head_size = 20

        glTranslatef(base_x, base_y, 0)

        glRotatef(self.camera_rotation[0], 1, 0, 0)
        glRotatef(self.camera_rotation[1], 0, 1, 0)

        scale_factor = 0.3
        glScalef(scale_factor, scale_factor, scale_factor)

        glLineWidth(2.0)
        glBegin(GL_LINES)

        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(arrow_length, 0, 0)

        glVertex3f(arrow_length, 0, 0)
        glVertex3f(arrow_length - arrow_head_size, arrow_head_size / 2, 0)
        glVertex3f(arrow_length, 0, 0)
        glVertex3f(arrow_length - arrow_head_size, -arrow_head_size / 2, 0)

        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, arrow_length, 0)

        glVertex3f(0, arrow_length, 0)
        glVertex3f(arrow_head_size / 2, arrow_length - arrow_head_size, 0)
        glVertex3f(0, arrow_length, 0)
        glVertex3f(-arrow_head_size / 2, arrow_length - arrow_head_size, 0)

        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, arrow_length)

        glVertex3f(0, 0, arrow_length)
        glVertex3f(arrow_head_size / 2, 0, arrow_length - arrow_head_size)
        glVertex3f(0, 0, arrow_length)
        glVertex3f(-arrow_head_size / 2, 0, arrow_length - arrow_head_size)

        glEnd()
        glLineWidth(1.0)

        glEnable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.imgui_renderer.shutdown()
                pygame.quit()
                return False

            self.imgui_renderer.process_event(event)

            if event.type == pygame.MOUSEMOTION:
                if not imgui.get_io().want_capture_mouse:
                    if event.buttons[0]:
                        self.camera_rotation[0] += event.rel[1]
                        self.camera_rotation[1] += event.rel[0]


            elif event.type == pygame.MOUSEBUTTONDOWN:
                if not imgui.get_io().want_capture_mouse:
                    if event.button == 4:  # Mouse wheel up
                        self.scale *= 1.1
                    elif event.button == 5:  # Mouse wheel down
                        self.scale *= 0.9

        return True

    def stop_simulation(self):
        self.simulating = False
        self.params = SimulationParameters()

        self.initial_points, self.control_points = self._initialize_control_points()
        self.X = self.control_points.reshape(64, 3)
        self.V = np.zeros(shape=(64, 3))

        self.control_frame = self.initial_frame.copy()

    def pause_simulation(self):
        self.simulating = False

    def start_simulation(self):
        self.simulating = True

    def draw_imgui(self):
        imgui.new_frame()

        size = self.controls_size_px

        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(size, self.display[1])

        imgui.begin("Controls", flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)

        if imgui.button("Start",  self.controls_size_px // 4, 25):
            self.start_simulation()

        imgui.same_line()
        if imgui.button("Pause", self.controls_size_px // 4, 25):
            self.pause_simulation()

        imgui.same_line()
        if imgui.button("Stop", self.controls_size_px // 4, 25):
            self.stop_simulation()

        # Simulation Parameters section
        imgui.text_colored("Simulation Parameters", 0.5, 0.7, 1.0)
        imgui.separator()

        _, self.params.h = imgui.input_float("h", self.params.h, 0.01)
        _, self.params.m = imgui.input_float("m", self.params.m, 1)
        _, self.params.c1 = imgui.input_float("c1", self.params.c1, 1)
        _, self.params.c2 = imgui.input_float("c2", self.params.c2, 1)
        _, self.params.k = imgui.input_float("k", self.params.k, 1)
        _, self.params.mu = imgui.input_float("mu", self.params.mu, 1)

        # Visualization Parameters section
        imgui.text_colored("Visualization Parameters", 0.5, 0.7, 1.0)
        imgui.separator()

        _, self.params.show_control_points = imgui.checkbox("Show Control Points", self.params.show_control_points)
        _, self.params.show_springs = imgui.checkbox("Show Springs", self.params.show_springs)
        _, self.params.show_control_frame = imgui.checkbox("Show Control Frame", self.params.show_control_frame)
        _, self.params.show_axes = imgui.checkbox("Show Axes", self.params.show_axes)
        _, self.params.show_faces = imgui.checkbox("Show Faces", self.params.show_faces)

        # Controls section
        imgui.text_colored("Control Parameters", 0.5, 0.7, 1.0)
        imgui.separator()

        x, y, z = self.params.translation
        _, translation = imgui.slider_float3("Translation", x, y, z, -1000, 1000)
        self.params.translation = np.array(translation)

        alpha, beta, gamma = self.params.rotation
        _, rotation = imgui.slider_float3("Rotation", alpha, beta, gamma, -180, 180)
        self.params.rotation = np.array(rotation)

        imgui.end()

        imgui.render()
        self.imgui_renderer.render(imgui.get_draw_data())

    def update_simulation(self):
        R = Rotation.from_euler("xyz", self.params.rotation, degrees=True).as_matrix()
        self.control_frame = self.initial_frame @ R + self.params.translation

        def spring_force(X, V, i, idx, l0, k, c, m):
            x1, x2 = X[i], X[idx]
            v1, v2 = V[i], V[idx]

            l = np.apply_along_axis(np.linalg.norm, axis=1, arr=x2 - x1)
            l_dot = ((x2 - x1) * (v2 - v1)).sum(axis=1) / l

            return (k * l_dot + c * (l - l0)) @ ((x2 - x1) / l[:, np.newaxis]) / m


        def derivatives(X, V):
            L = self.L
            neighbors = self.neighbors
            k = self.params.k
            c1 = self.params.c1
            c2 = self.params.c2
            m = self.params.m
            control_frame = self.control_frame
            
            X_dot = V
            V_dot = np.zeros_like(V)

            for i in range(len(V)):
                idx = neighbors[i, :]
                l0 = L[i, idx]
                V_dot[i] = spring_force(X, V, i, idx, l0, k, c1, m)

            x1, x2 = X[self.idx_cube], control_frame[self.idx_frame]
            v1, v2 = V[self.idx_cube], np.zeros_like(V[self.idx_cube])

            l0 = 0
            c = c2

            l = np.apply_along_axis(np.linalg.norm, axis=1, arr=x2 - x1)
            l_dot = ((x2 - x1) * (v2 - v1)).sum(axis=1) / (l + 1e-5)

            V_dot[self.idx_cube] += (k * l_dot + c * (l - l0))[:, np.newaxis] * ((x2 - x1) / (l[:, np.newaxis] + 1e-5)) / m

            return X_dot, V_dot

        # RK4
        X = self.X
        V = self.V
        h = self.params.h

        k1_x, k1_v = derivatives(X, V)

        X2 = X + (h / 2) * k1_x
        V2 = V + (h / 2) * k1_v
        k2_x, k2_v = derivatives(X2, V2)

        X3 = X + (h / 2) * k2_x
        V3 = V + (h / 2) * k2_v
        k3_x, k3_v = derivatives(X3, V3)

        X4 = X + h * k3_x
        V4 = V + h * k3_v
        k4_x, k4_v = derivatives(X4, V4)

        self.X += (h / 6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        self.V += (h / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        clipped_X = np.clip(self.X, -5, 5)
        mask = clipped_X == self.X
        self.V = np.where(mask, self.V, -self.params.mu * self.V)
        self.X = clipped_X

    def update_display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        glTranslatef(*self.camera_position)
        glRotatef(self.camera_rotation[0], 1, 0, 0)
        glRotatef(self.camera_rotation[1], 0, 1, 0)
        glScale(self.scale, self.scale, self.scale)

        if self.params.show_axes:
            self.draw_axes()

        self.draw_bounding_box()

        if self.params.show_control_frame:
            self.draw_control_frame()
        if self.params.show_control_points:
            self.draw_control_points()

        self.draw_imgui()

        pygame.display.flip()

        if self.simulating:
            self.update_simulation()


def main():
    simulation = BezierCubeSimulation()

    running = True
    while running:
        running = simulation.handle_input()
        if running:
            simulation.update_display()


if __name__ == "__main__":
    main()