# 5910406337 Phatchanya Chongsheveewat
import sys
from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import pandas as pd
import math as m
import time


win_w, win_h = 800, 600
model_filenames = ["models/bunny.tri"]
model_id = 0


def reshape(w, h):
    global win_w, win_h

    win_w, win_h = w, h
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, win_w/win_h, 0.01, 50)

is_specular_on = True
light_specular = [0.9, 0.3, 0.9, 1.0]
shininess = 5
def keyboard(key, x, y):
    global light_specular, is_specular_on, shininess

    key = key.decode("utf-8")
    if key == 's':
        is_specular_on = not is_specular_on
        if is_specular_on:
            light_specular = [0.9, 0.3, 0.9, 1.0]
        else:
            light_specular = [0.0, 0.0, 0.0, 1.0]
    elif key == 'n':
        shininess -= 1
        if shininess <= 0:
            shininess = 0
    elif key == 'N':
        shininess += 1
        if shininess > 128:
            shininess = 128
    elif key == 'q' or key == 'Q':
        exit(0)
    glutPostRedisplay()

tick = 0
def idle():
    global tick

    tick += 1
    glutPostRedisplay()

m = [0, 0, GLUT_LEFT_BUTTON, GLUT_UP]
eye_dist = [0, 0]
rotate_degree = [0, 0, 0]
def motion(x, y):
    dx, dy = x-m[0], y-m[1]
    button, state = m[2], m[3]
    m[0], m[1] = x, y
    if state == GLUT_DOWN:
        if button == GLUT_LEFT_BUTTON:
            if (glutGetModifiers() and GLUT_ACTIVE_SHIFT):
                if abs(dx) > abs(dy):
                    eye_dist[0] += (dx / 60)
                    if eye_dist[0] > 25:
                        eye_dist[0] = 25
                    elif eye_dist[0] < -25:
                        eye_dist[0] = -25
                else:
                    eye_dist[1] += (dy / 60)
                    if eye_dist[1] > 20:
                        eye_dist[1] = 20
                    elif eye_dist[1] < -20:
                        eye_dist[1] = -20
            else:
                if abs(dx) > abs(dy):
                    rotate_degree[0] += dx
                else:
                    rotate_degree[1] += dy
    glutPostRedisplay()

def mouse(button, state, x, y):
    m[0], m[1], m[2], m[3] = x, y, button, state
    glutPostRedisplay()

# def pan(x, y):
#     global pan_x, pan_y
#     if x < 0 or x > win_w: return
#     if y < 0 or y > win_h: return
#     x = 2*(x*1.0/win_h) - 1
#     y = 2*((win_h-y)*1.0/win_h) - 1
#     move = 0
#     if abs(x) - abs(y) > 0:
#         if x - pan_x > 0:
#             move = -0.1
#         elif x - pan_x < 0:
#             move = 0.1
#         pan_x = x
#     elif abs(x) - abs(y) < 0:
#         if y - pan_y > 0:
#             move = -0.1
#         elif y - pan_y < 0:
#             move = 0.1
#         pan_y = y

# zoom_y = 0
# def zoom(x, y):
#     global zoom_y
#     if x < 0 or x > win_w: return
#     if y < 0 or y > win_h: return
#     x = 2*(x*1.0/win_h) - 1
#     y = 2*((win_h-y)*1.0/win_h) - 1
#     zoom = 0
#     if y - zoom_y > 0:
#         zoom = -0.1
#     elif y - zoom_y < 0:
#         zoom = 0.1
#     zoom_y = y

def display():
    global positions, colors

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    centroid = models[model_id]["centroid"]
    bbox     = models[model_id]["bbox"]
    vertices = models[model_id]["vertices"]
    normals  = models[model_id]["normals"]
    colors   = models[model_id]["colors"]

    eye_pos = np.array((centroid[0], centroid[1], centroid[2]+1.5*bbox[0]), dtype=np.float32)
    gluLookAt(*eye_pos, *centroid, 0, 1, 0)

    light_ambient = [0.2, 0.2, 0.2, 1.0]
    light_diffuse = [0.1, 0.1, 0.1, 1.0]

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glMaterialfv(GL_FRONT, GL_AMBIENT, light_ambient)
    glMaterialfv(GL_FRONT, GL_DIFFUSE, light_diffuse)
    glMaterialfv(GL_FRONT, GL_SPECULAR, light_specular)
    glMaterialfv(GL_FRONT, GL_SHININESS, [shininess])

    glPushMatrix()
    glRotatef(rotate_degree[0], 0, 1, 0)
    glRotatef(rotate_degree[1], 1, 0, 0)
    glRotatef(rotate_degree[2], 0, 0, 1)

    glVertexPointer(3, GL_FLOAT, 0, vertices)
    glNormalPointer(GL_FLOAT, 0, normals)
    glColorPointer(3, GL_FLOAT, 0, colors)
    glDrawArrays(GL_TRIANGLES, 0, len(vertices))
    glPopMatrix()
    glutSwapBuffers()

models = []
def gl_init_models():
    global start_time

    glClearColor(0, 0, 0, 0)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    for i in range(len(model_filenames)):
        df = pd.read_csv(model_filenames[i], delim_whitespace=True, comment='#', header=None)
        centroid = df.values[:, 0:3].mean(axis=0)
        bbox = df.values[:, 0:3].max(axis=0) - df.values[:, 0:3].min(axis=0)

        vertices = np.ones((len(df.values), 3), dtype=np.float32)
        normals = np.zeros((len(df.values), 3), dtype=np.float32)
        vertices[:, 0:3] = df.values[:, 0:3]
        normals[:, 0:3] = df.values[:, 3:6]
        colors = 0.5*(df.values[:, 3:6].astype(np.float32) + 1)
        models.append({"vertices": vertices, "normals": normals, "colors": colors, 
                       "centroid": centroid, "bbox": bbox})
    start_time = time.time() - 0.0001

def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(win_w, win_h)
    glutCreateWindow(b"5910406337 Phatchanya Chongsheveewat")
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutPassiveMotionFunc(motion)
    glutMotionFunc(motion)
    glutMouseFunc(mouse)
    # glutZoomFunc(zoom)
    glutIdleFunc(idle)
    gl_init_models()
    glutMainLoop()

if __name__ == "__main__":
    main()
