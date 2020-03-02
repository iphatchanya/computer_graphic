import sys
from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *
from ctypes import c_void_p
import numpy as np
import pandas as pd
import math as m
import time

win_w, win_h = 1024, 768
model_filenames = ["models/bunny.tri", "models/horse.tri", "models/alien.tri"]
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
    global wireframe, pause

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
    vertices = models[model_id]["vertices"]
    normals  = models[model_id]["normals"]
    colors   = models[model_id]["colors"]
    centroid = models[model_id]["centroid"]
    centroid = models[model_id]["centroid"]
    bbox     = models[model_id]["bbox"]
    max_len  = max(bbox)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(centroid[0], centroid[1], centroid[2]+1.2*max_len, 
            *centroid, 0, 1, 0)
    glBindVertexArray(vao[0])
    glPushMatrix()
    glScalef(0.2, 0.2, 0.2)
    glTranslatef(-10, 0, 0)
    glDrawArrays(GL_TRIANGLES, 0, len(models[0]["vertices"]))
    glPopMatrix()
    glBindVertexArray(vao[1])
    glDrawArrays(GL_TRIANGLES, 0, len(models[1]["vertices"])) #mid = (tick//180) % 2
    glBindVertexArray(vao[2])
    glPushMatrix()
    glScalef(0.1, 0.1, 0.1)
    glTranslatef(20, 0, 5)
    glDrawArrays(GL_TRIANGLES, 0, len(models[2]["vertices"]))
    glPopMatrix()
    glutSwapBuffers()

models = []
def gl_init_models():
    global start_time, vao

    glClearColor(0.95, 0.95, 0.8, 0)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)

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

    vao = glGenVertexArrays(len(models))
    for i in range(len(models)):
        vertices = models[i]["vertices"]
        normals  = models[i]["normals"]
        colors   = models[i]["colors"]

        glBindVertexArray(vao[i]) #glBindVertexArray(None) -> ยกเลิกการ bind
        vbo = glGenBuffers(3)
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0])
        glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, vbo[1])
        glBufferData(GL_ARRAY_BUFFER, normals, GL_STATIC_DRAW)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(2)
        glBindBuffer(GL_ARRAY_BUFFER, vbo[2])
        glBufferData(GL_ARRAY_BUFFER, colors, GL_STATIC_DRAW)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(3)

    start_time = time.time() - 0.0001
        
def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(win_w, win_h)
    glutCreateWindow(b"3D Transformations")
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutIdleFunc(idle)
    gl_init_models()
    glutMainLoop()

if __name__ == "__main__":
    main()

#def draw_model(model_id=0):
#     if model_id < 0 or model_id >= len(models):
#         return


# VBO(จุด สี ...) is a subset of VAO(ตัวม้า กระต่าย ...) 
# VBO: build -> bind -> use (สร้างได้หลายอัน ใช้ทีละอัน)
# VAO เก็บค่าตัวเลข หรือ ตัวเลขที่เป็นอาร์เรย์

# VBO - บอกว่าเดต้าอยู่ที่ไหน
# Stream - animation, read-write ควรเร็วใกล้เคียงกัน
# glVertextAttribPointer() -> เรามักไม่ normalized ในนี้ -> อาจมีปัญหากับค่าสี
