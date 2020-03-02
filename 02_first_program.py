import sys
from OpenGL.GL import *
from OpenGL.GLU import * #เป็นฟังก์ชันเสริมที่ช่วย openGL - แปลงระบบพิกัด
from OpenGL.GLUT import *
from math import sin, cos, pi

#gluOrtho2D(left, right, bottom, top) - เอาของใหม่ไปคูณอันเก่า พอขยับหน้าต่างก็จะเรียกทุกครั้ง ผลทับซ้อน มันเลยเล็กลงเรื่อยๆ ควรเรียกครั้งเดียว
#ตอนหน้าต่างเปลี่ยนแปลงขนาดและตอนที่เรียกครั้งแรกไม่ใช่ตอนเลือนไปมา
def reshape(width, height):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity() #ไม่หวังผลต่อเนื่อง มันจะเอาของใหม่ไปคูณกับ identity
    gluOrtho2D(-width//2, width//2, -height//2, height//2)
    #print("Windows is reshaped %d %d!" % (width,height))

def display():
    #gluOrtho2D(-2, 2, -2, 2)
    glClearColor(1.0, 1.0, 0.0, 0.0)
    glClear(GL_COLOR_BUFFER_BIT)
    #glColor3f(0.0, 1.0, 1.0)
    # --------- สี่เหลี่ยม ----------
    # glBegin(GL_POLYGON)
    # glVertex3f(-0.5, -0.5, 0.0)
    # glVertex3f(0.5, -0.5, 0.0)
    # glVertex3f(0.5, 0.5, 0.0)
    # glVertex3f(-0.5, 0.5, 0.0)
    # glEnd()
    # --------- สามเหลี่ยม ----------
    # glColor3f(0, 0, 1)    
    # glBegin(GL_TRIANGLES) #ทุก3 GL_TRIANGLES จะได้ 3 เหลี่ยม 1 อัน
    # glVertex3f(0, 1, 0)
    # glColor3f(1, 0, 0)
    # glVertex3f(1, -1, 0)
    # glColor3f(0, 1, 0)
    # glVertex3f(-1, -1, 0)
    # glEnd()
    # --------- วงกลม ----------
    # glColor3f(0, 0, 1)    
    # glBegin(GL_POINTS)
    # for i in range(10*360):
    #     theta = i/1800 * pi
    #     x = 300*cos(theta)
    #     y = 300*sin(theta)
    #     glVertex2f(x,y)
    # glEnd()
    # --------- สามเหลี่ยมรูปพัด ----------
    glColor3f(0, 0, 1)    
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(0,0)
    for i in range(361):
        theta = i/180 * pi
        x = 300*cos(theta)
        y = 300*sin(theta)
        glVertex2f(x,y)
    glEnd()
    glFlush()

def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE)
    glutInitWindowSize(800, 600) #กำหนดขนาดก่อนสร้าง
    glutCreateWindow(b"Program Template")
    glutReshapeFunc(reshape)     
    glutDisplayFunc(display)
   
    #glClearColor(1.0, 1.0, 0.0, 0.0) #จะได้เรียกแค่ทีเดียว กรณี ไม่มีการเปลี่ยนสีหน้าจอ
    glutMainLoop()

if __name__ == "__main__":
    main()