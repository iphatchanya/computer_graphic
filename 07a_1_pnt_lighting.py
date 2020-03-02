import sys
from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *

rot_x = 0

def reshape(w, h):
	glViewport(0, 0, w, h)	
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	gluPerspective(60, w/h, 0.1, 50)

def display():
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()
	gluLookAt(0, 2, -3, 0, 0, 0, 0, 1, 0)

	al = [0.2, 0.2, 0.2, 1.0]
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, al)

	light_ambient = [0.2, 0.2, 0.2, 1.0]
	light_diffuse = [1.0, 1.0, 1.0, 1.0]
	light_specular = [1.0, 1.0, 1.0, 1.0]
	light_position = [-1.0, 1.0, -1.0, 0.0]
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)
	glLightfv(GL_LIGHT0, GL_POSITION, light_position)

	light_position = [-1.0, 0.5, -2.0, 1.0]
	glLightfv(GL_LIGHT0, GL_POSITION, light_position)

	glRotatef(rot_x, 0, 1, 0)
	glutSolidTeapot(1)
	glutSwapBuffers()

def idle():
	global rot_x
	rot_x += 1
	glutPostRedisplay()

def gl_init():
	glClearColor(0, 0, 0, 0)
	glEnable(GL_DEPTH_TEST)
	glShadeModel(GL_SMOOTH)
	glEnable(GL_LIGHTING)
	glEnable(GL_LIGHT0)
	glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE)

def main():
	glutInit(sys.argv)
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
	glutInitWindowSize(1024, 768)
	glutCreateWindow(b"Point Light Source")
	glutDisplayFunc(display)
	glutReshapeFunc(reshape)
	glutIdleFunc(idle)
	gl_init()
	glutMainLoop()

if __name__ == "__main__":
	main()