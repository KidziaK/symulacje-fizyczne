from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np
import pygame
import imgui
import quaternion

from imgui.integrations.pygame import PygameRenderer
from scipy.spatial.transform import Rotation

def sp_r_to_np_q(r):
    """
    Scipy has a convention where real part of the quaternion is last, whereas numpy-quaternion package
    assumes real part to be the first component.
    """
    q = r.as_quat()
    return np.quaternion(q[3], q[0], q[1], q[2])

def rot_mat(q):
    return quaternion.as_rotation_matrix(q)

class CubeVisualization:
    def __init__(self):
        # Initialize Pygame and OpenGL
        pygame.init()
        self.display = (1500, 1000)  # Larger window to match reference
        self.window = pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Cube Simulation")

        # Initialize ImGui
        imgui.create_context()
        self.imgui_renderer = PygameRenderer()
        self.io = imgui.get_io()
        self.io.display_size = self.display[0], self.display[1]

        # Simulation parameters
        self.cube_edge_length = 2.0
        self.cube_density = 1.0
        self.cube_deviation = 15 # degrees
        self.angular_velocity = 15 # revolutions per second (360 * self.angular_velocity degrees per second)
        self.integration_step = 0.001
        self.path_length = 5000
        self.simulating = False

        # Display flags
        self.display_cube = True
        self.display_path = True
        self.display_diagonal = True
        self.display_plane = True
        self.gravity_on = True

        # Camera settings
        self.camera_distance = 10.0
        self.camera_rotation = [30, 45, 0]

        # OpenGL setup
        self.setup_gl()

        # Cube setup
        self.faces = np.array([
            [0, 1, 2, 3],  # Bottom face
            [4, 5, 6, 7],  # Top face
            [0, 1, 5, 4],  # Front face
            [1, 2, 6, 5],  # Right face
            [2, 3, 7, 6],  # Back face
            [3, 0, 4, 7],  # Left face
        ])

        self.original_vertices = self.get_initial_vertices()
        self.Q = self.get_initial_rotation()
        B = rot_mat(self.Q)
        self.vertices = (B @ self.original_vertices.T).T
        self.W = self.get_initial_angular_velocity()
        self.I = self.get_initial_inertia_tensor()

        # Misc
        self.plane_size = 2.0
        self.controls_size_px = 400
        self.path = np.zeros(shape=(self.path_length, 3))
        self.current_length = 0

    def get_initial_rotation(self):
        deviation_rad = np.deg2rad(self.cube_deviation)
        yx_angles = np.array([-np.pi / 4, np.arctan(np.sqrt(2) / 2) - np.pi / 2 + deviation_rad])
        r = Rotation.from_euler("yx", yx_angles)
        return sp_r_to_np_q(r)

    def get_initial_angular_velocity(self):
        radians = np.deg2rad(self.angular_velocity * 360)
        w = self.vertices[6]
        w = w / np.linalg.norm(w) * radians
        W = rot_mat(self.Q).T @ w
        return W

    def get_initial_inertia_tensor(self):
        X2 = Y2 = Z2 = (self.cube_edge_length * 5) / 3
        return np.diag([Y2 + Z2, X2 + Z2, X2 + Y2])

    def setup_gl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.0, 0.0, 0.0, 1.0)

    def draw_axes(self):
        glBegin(GL_LINES)

        # X axis - Red
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(5, 0, 0)

        # Y axis - Green
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 5, 0)

        # Z axis - Blue
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 5)

        glEnd()

    def draw_cube(self):
        B = rot_mat(self.Q)
        self.vertices = (B @ self.original_vertices.T).T
        self.path = np.roll(self.path, 1, axis=0)
        self.path[0] = self.vertices[6]
        self.current_length = min(self.current_length + 1, self.path_length)

        if not self.display_cube:
            return

        if self.display_diagonal:
            glBegin(GL_LINES)

            glColor3f(0.5, 0.5, 1.0)
            glVertex3f(*self.vertices[0])
            glVertex3f(*self.vertices[6])

            glEnd()

        glColor4f(0.5, 0.5, 1.0, 0.5)

        glBegin(GL_QUADS)

        for face in self.faces:
            for vertex in face:
                v = np.array(self.vertices[vertex])
                glVertex3fv(v)

        glEnd()

    def draw_bezier_curve(self):
        if self.current_length == 0:
            return

        glBegin(GL_LINE_STRIP)
        glColor3f(1, 1, 1)

        non_zero_points = self.path[:self.current_length]

        # Efficiently iterate through all points using numpy array operations
        np.apply_along_axis(lambda point: glVertex3f(*point), 1, non_zero_points)

        glEnd()

    def draw_ground_plane(self):
        if not self.display_plane:
            return

        glColor4f(0.5, 0.5, 0.5, 0.3)

        size = self.plane_size

        glBegin(GL_QUADS)
        glVertex3f(-size, 0, -size)
        glVertex3f(size, 0, -size)
        glVertex3f(size, 0, size)
        glVertex3f(-size, 0, size)

        glEnd()

    def start_simulation(self):
        self.simulating = True

    def stop_simulation(self):
        self.simulating = False

    def get_initial_vertices(self):
        return self.cube_edge_length * np.array([
            [0, 0, 0],  # Vertex 0
            [1, 0, 0],  # Vertex 1
            [1, 1, 0],  # Vertex 2
            [0, 1, 0],  # Vertex 3
            [0, 0, 1],  # Vertex 4
            [1, 0, 1],  # Vertex 5
            [1, 1, 1],  # Vertex 6
            [0, 1, 1],  # Vertex 7
        ])

    def reset_simulation(self):
        self.original_vertices = self.get_initial_vertices()
        self.Q = self.get_initial_rotation()
        B = rot_mat(self.Q)
        self.vertices = (B @ self.original_vertices.T).T
        self.W = self.get_initial_angular_velocity()
        self.I = self.get_initial_inertia_tensor()
        self.path = np.zeros(shape=(self.path_length, 3))
        self.current_length = 0

    def apply_conditions(self):
        self.reset_simulation()

    def draw_gui(self):
        imgui.new_frame()

        size = self.controls_size_px

        imgui.set_next_window_position(self.display[0] - size, 0)
        imgui.set_next_window_size(size, self.display[1])

        imgui.begin("Controls", flags=imgui.WINDOW_NO_RESIZE|imgui.WINDOW_NO_MOVE)

        if imgui.button("Start Simulation", 130, 25):
            self.start_simulation()

        imgui.same_line()
        if imgui.button("Stop Simulation", 130, 25):
            self.stop_simulation()

        # Initial Conditions section
        imgui.text_colored("Initial Conditions", 0.5, 0.7, 1.0)
        imgui.separator()

        _, self.cube_edge_length = imgui.input_float("Cube Edge Length", self.cube_edge_length, 0.1)
        _, self.cube_density = imgui.input_float("Cube Density", self.cube_density, 0.1)
        _, self.cube_deviation = imgui.input_float("Cube Deviation", self.cube_deviation, 0.1)
        _, self.angular_velocity = imgui.input_float("Angular Velocity", self.angular_velocity, 0.1)
        _, self.integration_step = imgui.input_float("Integration Step", self.integration_step, 0.0001, format="%.6f")

        if imgui.button("Apply Conditions"):
            self.apply_conditions()

        imgui.dummy(0, 10)

        # Visualization section
        imgui.text_colored("Visualization", 0.5, 0.7, 1.0)
        imgui.separator()

        _, self.path_length = imgui.input_int("Path length", self.path_length)
        _, self.display_cube = imgui.checkbox("Display Cube", self.display_cube)
        _, self.display_path = imgui.checkbox("Display Path", self.display_path)
        _, self.display_diagonal = imgui.checkbox("Display Diagonal", self.display_diagonal)
        _, self.display_plane = imgui.checkbox("Display Plane", self.display_plane)

        imgui.dummy(0, 10)

        # Other section
        imgui.text_colored("Other", 0.5, 0.7, 1.0)
        imgui.separator()
        _, self.gravity_on = imgui.checkbox("Gravity On", self.gravity_on)

        imgui.end()

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Set up camera
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -self.camera_distance)
        glRotatef(self.camera_rotation[0], 1, 0, 0)
        glRotatef(self.camera_rotation[1], 0, 1, 0)
        glRotatef(self.camera_rotation[2], 0, 0, 1)

        # Draw scene elements
        self.draw_axes()
        self.draw_ground_plane()
        self.draw_cube()
        self.draw_bezier_curve()

        # Render ImGui
        self.draw_gui()
        imgui.render()
        self.imgui_renderer.render(imgui.get_draw_data())

        pygame.display.flip()

        # Update Simulation
        if self.simulating:
            self.update_simulation()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            self.imgui_renderer.process_event(event)

            if event.type == pygame.MOUSEMOTION:
                if event.buttons[0] and not imgui.get_io().want_capture_mouse:
                    self.camera_rotation[1] += event.rel[0]
                    self.camera_rotation[0] += event.rel[1]

        return True

    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running:
            running = self.handle_input()
            self.render()
            clock.tick(120)

        self.imgui_renderer.shutdown()
        pygame.quit()

    def cube_mass(self):
        return (self.cube_edge_length ** 3) * self.cube_density

    def update_simulation(self):

        def derivatives(Q, W):
            B = rot_mat(self.Q)

            f = np.array([0, -9.81 * self.cube_mass(), 0]) if self.gravity_on else np.zeros(3)
            cross_with_f = lambda r: np.cross(r, f)
            r = self.vertices
            n = np.apply_along_axis(cross_with_f, axis=1, arr=r).sum(axis=0)
            N = B.T @ n

            W_dot = np.linalg.inv(self.I) @ N + np.linalg.inv(self.I) @ np.cross(self.I @ W, W)
            Q_dot = 0.5 * Q * np.quaternion(0, *W)

            return W_dot, Q_dot

        # RK4
        h = self.integration_step
        W = self.W
        Q = self.Q

        k1_w, k1_q = derivatives(Q, W)

        Q2 = Q + (h / 2) * k1_q
        W2 = W + (h / 2) * k1_w
        k2_w, k2_q = derivatives(Q2, W2)

        Q3 = Q + (h / 2) * k2_q
        W3 = W + (h / 2) * k2_w
        k3_w, k3_q = derivatives(Q3, W3)

        Q4 = Q + h * k3_q
        W4 = W + h * k3_w
        k4_w, k4_q = derivatives(Q4, W4)

        self.W += (h / 6) * (k1_w + 2 * k2_w + 2 * k3_w + k4_w)
        self.Q += (h / 6) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)

        self.Q = self.Q.normalized()


if __name__ == "__main__":
    visualization = CubeVisualization()
    visualization.run()
