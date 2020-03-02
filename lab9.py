import sys
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import pandas as pd
import math as m
from ctypes import c_void_p

from PIL import Image

def load_texture(filename):
    try:
        im = Image.open(filename)
    except:
        print("Error:", sys.exc_info()[0])
    w = im.size[0]
    h = im.size[1]
    image = im.tobytes("raw", "RGB", 0)
    tex_id = glGenTextures(1)
    glActiveTexture(GL_TEXTURE0) # use texture unit number 0
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, 3, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, image)

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
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 800/600, 0.1, 30)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(0, 0, 20, 0, 5, 0, 0, 1, 0)
    glColor3f(0, 0, 1)
    glDrawArrays(GL_TRIANGLES, 0, n_vertices) # glutSolidTeapot(1) - วาดกาน้ำ
    glutSwapBuffers()

def init():
    vert_id = glCreateShader(GL_VERTEX_SHADER)
    frag_id = glCreateShader(GL_FRAGMENT_SHADER)

    vert_code = b'''
#version 120
attribute vec3 position;
attribute vec3 normal;
attribute vec2 uv;
varying vec3 f_color;
varying vec2 f_uv;
void main()
{
    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vec4(position, 1);
    f_color = gl_Color.rgb;
    f_uv = uv;
}'''

    frag_code = b'''
#version 120
uniform sampler2D demon_texture;
varying vec3 f_color;
varying vec2 f_uv;
void main()
{
    gl_FragColor = texture2D(demon_texture, 10 * f_uv);
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

    try:
        df = pd.read_csv("models/bunny.tri", delim_whitespace=True, comment='#', header=None)
    except:
        print("%s not found!" % "bunny.tri")
        exit(1)

    globals()["n_vertices"] = n_vertices =  len(df.values)
    positions = np.zeros((n_vertices, 3), dtype=np.float32) #array 
    normals = np.zeros((n_vertices, 3), dtype=np.float32)
    uvs = np.zeros((n_vertices, 2), dtype=np.float32) # เป็น float 32
    positions[:, 0:3] = df.values[:, 0:3]
    normals[:, 0:3] = df.values[:, 3:6]
    uvs[:, 0:2] = df.values[:, 6:8] #(*)uvs = df.values[:, 6:8] เป็น float64 (ค่าdefault)
    print("Loaded %d vertices" % n_vertices)

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao) #จะใช้งานก็ bind เข้าไป

    vbo = glGenBuffers(3)
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0])
    glBufferData(GL_ARRAY_BUFFER, positions, GL_STATIC_DRAW)
    position_loc = glGetAttribLocation(prog_id, "position")
    glVertexAttribPointer(position_loc, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
    glEnableVertexAttribArray(position_loc)

    glBindBuffer(GL_ARRAY_BUFFER, vbo[1])
    glBufferData(GL_ARRAY_BUFFER, normals, GL_STATIC_DRAW)
    normal_loc = glGetAttribLocation(prog_id, "normal")
    if normal_loc != -1:
        glVertexAttribPointer(normal_loc, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(normal_loc)    

    glBindBuffer(GL_ARRAY_BUFFER, vbo[2])
    glBufferData(GL_ARRAY_BUFFER, uvs, GL_STATIC_DRAW)
    uv_loc = glGetAttribLocation(prog_id, "uv")
    if uv_loc != -1 :
        glVertexAttribPointer(uv_loc, 2, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(uv_loc)    

    load_texture("texture_map/demon.png")
    demon_texture_loc = glGetUniformLocation(prog_id, "demon_texture") #ทำการเชื่อมต่อ เอาโลเคชันมาจาก GLSL
    glUniform1i(demon_texture_loc, 0) #เชื่อมต่อรูปกับสิ่งที่โหลด
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_DEPTH_TEST)

def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutInitWindowPosition(50, 50)    
    glutCreateWindow(b"GLSL")
    glutDisplayFunc(display)
    init()
    glutMainLoop()

if __name__ == "__main__":
    main()
# Data Type Classifier:
# varying vec3 x;
# uniform vec3 y;
# attribute vec3 z;
    # ลักษณะเฉพาะของ vertex (ตำแหน่ง ค่าสี ...)

# การบ้านนนนนนนนนนนนน
    # glutModifier พวกคีย์พิเศษ(shift, alt, ...)

