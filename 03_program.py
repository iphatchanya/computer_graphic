import sys
from OpenGL.GL import *
from OpenGL.GLU import * #เป็นฟังก์ชันเสริมที่ช่วย openGL - แปลงระบบพิกัด
from OpenGL.GLUT import *
import numpy as np
import pandas as pd
tx, ty = 0, 0
df = None
animation_on = False # เก็บสถานะ

def display():
    glClearColor(1, 1, 0, 0)
    # glClear(GL_COLOR_BUFFER_BIT)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glColor3f(0, 0, 1)    
    # glBegin(GL_TRIANGLE_FAN) #ทุก3 GL_TRIANGLES จะได้ 3 เหลี่ยม 1 อัน
    # glVertex2f(-0.5+tx, -0.5+ty)
    # glVertex2f(0+tx, 0.5+ty)
    # glVertex2f(0.5+tx, -0.5+ty)
    # glEnd()
    #-------------------------------
    glBegin(GL_TRIANGLES)
    # for line in df.values[:, 0:3]:
    #     glColor3f(line[3], line[4], line[5])
    #     glVertex3f(line[0], line[1], line[2])
    for line in df.values[:, 0:6]:
        glColor3dv(line[3:6])
        glVertex3dv(line[0:3])
    glEnd()
    glutSwapBuffers()
    # glFlush()

def idle():
    glRotatef(1, 0, 0, 1)
    glutPostRedisplay()

def keyboard(key, x, y):
    global tx, ty, animation_on # ทำให้เป็น global
    key = key.decode("utf-8")
    if key == 'i' : ty = ty + 0.1
    elif key == 'k' : ty = ty - 0.1 #ty แรกเป็น local ty ตัวที่สองเป็น global
    elif key == 'j' : tx = tx - 0.1
    elif key == 'l' : tx = tx + 0.1
    elif key == ' ' : animation_on = not animation_on
    elif key == 'q' : exit()
    glutIdleFunc(idle if animation_on else None)
    # if animation_on :
    #     glutIdleFunc(idle)
    # else:
    #     glutIdleFunc(None) #ยกเลิกการผูก idle
    glutPostRedisplay() #บังคับให้วาดเลย

def main():
    global df
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(800, 600) #กำหนดขนาดก่อนสร้าง
    glutCreateWindow(b"Program Template")  
    glutDisplayFunc(display)
    glutKeyboardFunc(keyboard)
    df = pd.read_csv("monkey.tri", delim_whitespace=True, comment='#', header=None)
    glOrtho(-1.5, 1.5, -1.5, 1.5, -1.5, 1.5)
    glEnable(GL_DEPTH_TEST)
    glutMainLoop()

if __name__ == "__main__":
    main()