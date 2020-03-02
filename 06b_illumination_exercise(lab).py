import sys
from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import pandas as pd
import math as m
import time

win_w, win_h = 1024, 768
model_filenames = ["models/bunny.tri", "models/horse.tri"]
model_id = 0

def reshape(w, h):
    global win_w, win_h

    win_w, win_h = w, h
    glViewport(0, 0, w, h)  
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, win_w/win_h, 0.01, 50)

wireframe, pause = False, True
def keyboard(key, x, y):
    global wireframe, pause, n

    key = key.decode("utf-8")
    if key == ' ':
        pause = not pause
        glutIdleFunc(None if pause else idle)
    elif key == 'w':
        wireframe = not wireframe
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if wireframe else GL_FILL)
    elif key == 'q':
        exit(0)
    glutPostRedisplay()

tick = 0
def idle():
    global tick

    tick += 1
    glutPostRedisplay()

def display():
    print("%.2f fps" % (tick/(time.time()-start_time)), tick, end='\r')      
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
    light_pos = eye_pos
    
    # การคำนวณแบบ vector ใช้ numpy
    Ia = np.array((1, 1, 1), dtype=np.float32)
    Id = np.array((1, 1, 1), dtype=np.float32)
    Is = np.array((1, 1, 1), dtype=np.float32)
    Ka = np.array((0.01, 0.01, 0.01), dtype=np.float32)
    Kd = np.array((0.0, 0.8, 0.8), dtype=np.float32)
    Ks = np.array((1, 1, 0), dtype=np.float32)    
    n = 50
    light_pos = np.array((3+20*m.cos(tick/180*m.pi), 4, 0+20*m.sin(tick/180*m.pi)))

    ambient = Ka * Ia
    L = light_pos - vertices
    L = L * 1/np.linalg.norm(L, axis=1).reshape(-1, 1)
    N = normals
    NdotL = np.sum(np.multiply(N, L), axis=1)
    diffuse = Kd * np.maximum(0, NdotL).reshape(-1, 1) * Id
    V = eye_pos = vertices
    V = V * 1/np.linalg.norm(V, axis=1).reshape(-1, 1)
    R = -L +2 * NdotL.reshape(-1, 1) * N
    VdotR = np.sum(np.multiply(V, R), axis=1)
    specular = Ks * np.power(np.maximum(0, VdotR), n).reshape(-1, 1) * Is
    colors[:] = ambient + diffuse + specular

    glVertexPointer(3, GL_FLOAT, 0, vertices)
    glNormalPointer(GL_FLOAT, 0, normals)
    glColorPointer(3, GL_FLOAT, 0, colors)
    glDrawArrays(GL_TRIANGLES, 0, len(vertices))
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
        print("Model: %s, no. of vertices: %d, no. of triangles: %d" % 
               (model_filenames[i], len(vertices), len(vertices)//3))
        print("Centroid:", centroid)
        print("BBox:", bbox)
    start_time = time.time() - 0.0001
    
def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(win_w, win_h)
    glutCreateWindow(b"Illumination")
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutIdleFunc(idle)
    gl_init_models()
    glutMainLoop()

if __name__ == "__main__":
    main()