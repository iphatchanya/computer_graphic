import sys
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from math import *
import math as m

def print_shader_info_log(shader, prompt=""):
    result = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not result:
        print("%s: %s" % (prompt, glGetShaderInfoLog(shader).decode("utf-8")))
        exit()

def print_program_info_log(shader, prompt=""):
    result = glGetProgramiv(shader, GL_LINK_STATUS)
    if not result:
        print("%s: %s" % (prompt, glGetProgramInfoLog(shader).decode("utf-8")))
        exit()

def display():
    glClearColor(1, 1, 1, 1)
    glClear(GL_COLOR_BUFFER_BIT)
    glColor3f(0,0,1)
    glutSolidTeapot(1)
    glutSwapBuffers()

cnt = 0
def idle():
    global cnt
    cnt = cnt + 1
    glUniform1f(my_cpu_input1_location, 2 * m.sin(cnt/180*m.pi))
    glutPostRedisplay()


def init():
    global my_cpu_input1_location
    vert_id = glCreateShader(GL_VERTEX_SHADER)
    frag_id = glCreateShader(GL_FRAGMENT_SHADER)

    vert_code = b'''   
#version 110
uniform float my_cpu_input1;
varying vec3 fcolor;
void main()
{
    float angle = my_cpu_input1 * length(gl_Vertex.xy);
    float s = sin(angle);
    float c = cos(angle);
    gl_Position.x = c * gl_Vertex.x - s * gl_Vertex.y;
    gl_Position.y = s * gl_Vertex.x + c * gl_Vertex.y;
    gl_Position.w = 1.0;
    fcolor = 0.5 * (gl_Normal + 1.0);
}'''

    frag_code = b'''
#version 110
uniform vec3 my_cpu_input2;
varying vec3 fcolor;
void main()
{
    gl_FragColor = vec4(fcolor, 1);
}'''

    glShaderSource(vert_id, vert_code)
    glShaderSource(frag_id, frag_code)

    glCompileShader(vert_id)
    glCompileShader(frag_id)
    print_shader_info_log(vert_id, "Vertex Shader")
    print_shader_info_log(frag_id, "Fragment Shader")

    prog_id = glCreateProgram()
    glAttachShader(prog_id, vert_id)
    glAttachShader(prog_id, frag_id)

    glLinkProgram(prog_id)
    print_program_info_log(prog_id, "Link Program")
    glUseProgram(prog_id)


    my_cpu_input1_location = glGetUniformLocation(prog_id, "my_cpu_input1")
    my_cpu_input2_location = glGetUniformLocation(prog_id, "my_cpu_input2")

    glUniform1f(my_cpu_input1_location, 1.5)
    glUniform3f(my_cpu_input2_location, 1.0, 0.25, 0.25)

def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(512, 512)
    glutInitWindowPosition(50, 50)    
    glutCreateWindow(b"GLSL")
    glutDisplayFunc(display)
    init()
    glutIdleFunc(idle)
    glutMainLoop()

if __name__ == "__main__":
    main()