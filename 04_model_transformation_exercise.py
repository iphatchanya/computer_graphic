import sys
from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import pandas as pd
import math as m
import time
import gl_helpers
win_w, win_h = 1024, 768
df = pd.read_table("models/bunny.tri", delim_whitespace=True, comment='#', header=None)
centroid = df.values[:, 0:3].mean(axis=0)

def reshape(w, h):
    global win_w, win_h

    win_w, win_h = w, h
    glViewport(0, 0, w, h)  
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, win_w/win_h, 0.01, 50)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(centroid[0], centroid[1], centroid[2]+18, *centroid, 0, 1, 0)
    # gluOrtho2D(-win_w//2, win_w//2, -win_h//2, win_h//2)

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

xx, yy = 0, 0
def motion(x, y):
    global xx, yy
    xx, yy = x, y

def display():
    global positions, colors

    print("%.2f fps" % (tick/(time.time()-start_time)), tick, end='\r', file=sys.stdout, flush=True)
    # a = abs(5*m.cos(3 * tick * m.pi / 180))
    # b = abs(5*m.sin(3 * tick * m.pi / 180))

    # type W แสดงเส้นโครงสร้าง
    a = m.cos(3 * tick * m.pi / 180)
    b = m.sin(3 * tick * m.pi / 180)
    mat = np.array([[1, 0, 0, 10*(xx-512)/512],
                    [0, 1, 0, -10*(yy-384)/384],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=np.float32)
    x_positions = ((mat @ gl_helpers.Rotate(tick, 0, 1, 0)) @ positions.T).T 
    # x_positions = (gl_helpers.Rotate(tick, 0, 1, 0) @ positions.T).T 
    # x_positions = (mat @ positions.T).T #สองแสนชุด ชุดละ 4 ค่า ใช้ความสามารถของ cpu (cpu จะยุ่งมาก) อนาคตจะส่งให้การ์ดจอทำ

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
  
    glVertexPointer(3, GL_FLOAT, 0, x_positions[:, 0:3]) #เอาตัวที่ 0 1 2 ไม่เอา 3
    glColorPointer(3, GL_FLOAT, 0, colors)
    glDrawArrays(GL_TRIANGLES, 0, len(x_positions)) 

    # glBegin(GL_TRIANGLES)
    # for i in range(len(x_positions)): #ปัญหาคือช้าเพราะลูปสองแสนกว่าครั้ง
    #     glColor3fv(colors[i])
    #     glVertex3fv(x_positions[i][0:3])
    # glEnd()

    glutSwapBuffers()

def gl_init():
    global positions, normals, colors
    global start_time

    glEnableClientState(GL_VERTEX_ARRAY) #จะได้เรียกแค่ครั้งเดียว
    glEnableClientState(GL_COLOR_ARRAY)
    glClearColor(0.95, 0.95, 0.8, 0)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    positions = np.ones((len(df.values), 4), dtype=np.float32) #ทำให้ทั้ง 4 ตัวมี่ค่าเป็น 1
    normals = np.zeros((len(df.values), 4), dtype=np.float32)
    positions[:, 0:3] = df.values[:, 0:3]
    normals[:, 0:3] = df.values[:, 3:6]
    colors = 0.5*(df.values[:, 3:6].astype(np.float32) + 1)

    start_time = time.time() - 0.0001 #คำนวณค่า framerate (1 วิได้กี่เฟรม - สูงดี)
    
def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(win_w, win_h)
    glutCreateWindow(b"Transformations")
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMotionFunc(motion)
    glutIdleFunc(idle)
    gl_init()
    glutMainLoop()

if __name__ == "__main__":
    main()
