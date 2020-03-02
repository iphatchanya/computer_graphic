import sys
from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import pandas as pd
from PIL import Image
import gl_helpers as glh

win_w, win_h = 1024, 768
t_value, wireframe, pause = 0, False, True
n_vertices, positions, colors, normals, uvs = 0, None, None, None, None
centroid, bbox = None, None
mouse = [0, 0, GLUT_LEFT_BUTTON, GLUT_UP]
rotate_degree = [0, 0, 0]

def motion_func(x ,y):
    dx, dy = x-mouse[0], y-mouse[1]
    button, state = mouse[2], mouse[3]
    mouse[0], mouse[1] = x, y
    if state == GLUT_DOWN:
        if button == GLUT_LEFT_BUTTON:
            if abs(dx) > abs(dy):
                rotate_degree[0] += dx
            else:
                rotate_degree[1] += dy
        elif button == GLUT_MIDDLE_BUTTON:
            if abs(dx) > abs(dy):
                rotate_degree[2] += dx
            else:
                rotate_degree[2] += dy
    glutPostRedisplay()

def mouse_func(button, state, x, y):
    mouse[0], mouse[1], mouse[2], mouse[3] = x, y, button, state
    glutPostRedisplay()

def reshape(w, h):
    global win_w, win_h, proj_mat

    win_w, win_h = w, h
    glViewport(0, 0, w, h)  
    proj_mat = glh.Perspective(60, win_w/win_h, 0.01, 10)

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

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    view_mat = glh.LookAt(centroid[0], centroid[1], centroid[2]+1.2*max(bbox), *centroid, 0, 1, 0)
    model_mat = glh.Rotate(rotate_degree[0], 0, 1, 0)
    model_mat = model_mat @ glh.Rotate(rotate_degree[1], 1, 0, 0)
    model_mat = model_mat @ glh.Rotate(rotate_degree[2], 0, 0, 1)

    glUniformMatrix4fv(glGetUniformLocation(prog_id, "model_mat"), 1, True, model_mat)
    glUniformMatrix4fv(glGetUniformLocation(prog_id, "view_mat"), 1, True, view_mat)
    glUniformMatrix4fv(glGetUniformLocation(prog_id, "proj_mat"), 1, True, proj_mat)
    glUniform3fv(glGetUniformLocation(prog_id, "light_pos"), 1, [-2, 2, 2])
    glUniform3fv(glGetUniformLocation(prog_id, "Ka"), 1, [0.01, 0.01, 0.01])
    glUniform3fv(glGetUniformLocation(prog_id, "Ks"), 1, [1, 1, 0])
    glUniform3fv(glGetUniformLocation(prog_id, "light_intensity"), 1, [1, 1, 1])
    glUniform1f(glGetUniformLocation(prog_id, "shininess"), 30)
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLES, 0, n_vertices)
    glutSwapBuffers()

t_value = 0
def idle():
    global t_value
    t_value += 0.01
    glutPostRedisplay()

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

def load_texture(filename, texture_unit=GL_TEXTURE0):
    try:
        im = Image.open(filename)
    except:
        print("Error:", sys.exc_info()[0])
        raise  
    w = im.size[0]
    h = im.size[1]
    image = im.tobytes("raw", "RGB", 0)
    tex_id = glGenTextures(1)
    glActiveTexture(texture_unit)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, 3, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, image)

def init_shaders():
    global prog_id
    global vao, vbo

    vert_id = glCreateShader(GL_VERTEX_SHADER)
    frag_id = glCreateShader(GL_FRAGMENT_SHADER)

    vert_code = b'''
#version 140
attribute vec3 position, color, normal;
attribute vec2 uv;
uniform mat4 model_mat, view_mat, proj_mat;
varying vec3 v_position, v_normal, eye_pos;
varying vec2 v_uv;
void main()
{
    gl_Position = proj_mat * view_mat * model_mat * vec4(position, 1);
    v_position = (model_mat * vec4(position, 1)).xyz;
    v_normal = (transpose(inverse(model_mat)) * vec4(normal, 0)).xyz;
    eye_pos = (inverse(view_mat) * vec4(0, 0, 0, 1)).xyz;
    v_uv = uv;
}
'''
    frag_code = b'''
#version 130
uniform sampler2D bunny_map;
uniform sampler2D brick_map;
uniform vec3 light_pos, light_intensity, Ka, Ks;
uniform float shininess;
varying vec3 v_position, v_normal, eye_pos;
varying vec2 v_uv;
void main()
{
    vec3 Kd = mix(texture(bunny_map, v_uv), 
                texture(brick_map, 10*v_uv), 0.5).rgb;
    vec3 L = normalize(light_pos - v_position);
    vec3 l1 = Ka * light_intensity;
    vec3 N = normalize(v_normal);
    vec3 l2 = Kd * max(abs(dot(N, L)), 0) * light_intensity;
    vec3 V = normalize(eye_pos - v_position);
    vec3 R = -L + 2*max(dot(N, L), 0)*N;
    vec3 l3 = Ks * pow(max(dot(V, R), 0), shininess) * light_intensity;
    gl_FragColor.rgb = l1 + l2 + l3;
}
'''

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
    print_program_info_log(prog_id, "Link error")

    glUseProgram(prog_id)

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(4)
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0])
    glBufferData(GL_ARRAY_BUFFER, positions, GL_STATIC_DRAW)
    position_loc = glGetAttribLocation(prog_id, "position")
    if position_loc != -1:
        glVertexAttribPointer(position_loc, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(position_loc)

    color_loc = glGetAttribLocation(prog_id, "color")
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1])
    glBufferData(GL_ARRAY_BUFFER, colors, GL_STATIC_DRAW)
    if color_loc != -1:
        glVertexAttribPointer(color_loc, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(color_loc)

    normal_loc = glGetAttribLocation(prog_id, "normal")
    glBindBuffer(GL_ARRAY_BUFFER, vbo[2])
    glBufferData(GL_ARRAY_BUFFER, normals, GL_STATIC_DRAW)
    if normal_loc != -1:
        glVertexAttribPointer(normal_loc, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(normal_loc)

    uv_loc = glGetAttribLocation(prog_id, "uv")
    glBindBuffer(GL_ARRAY_BUFFER, vbo[3])
    glBufferData(GL_ARRAY_BUFFER, uvs, GL_STATIC_DRAW)
    if uv_loc != -1:
        glVertexAttribPointer(uv_loc, 2, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(uv_loc)

    load_texture("texture_map/bunny_hair.jpg", GL_TEXTURE0)
    load_texture("texture_map/brick_wall_small.jpg", GL_TEXTURE1)

    texture_map_location = glGetUniformLocation(prog_id, "bunny_map")
    glUniform1i(texture_map_location, 0)
    texture_map_location = glGetUniformLocation(prog_id, "brick_map")
    glUniform1i(texture_map_location, 1)

def init_gl_and_model():
    global n_vertices, positions, colors, normals, uvs, centroid, bbox

    glClearColor(0.01, 0.01, 0.2, 0)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)

    df = pd.read_csv("models/bunny_uv.tri", delim_whitespace=True, comment='#',
                     header=None, dtype=np.float32)
    centroid = df.values[:, 0:3].mean(axis=0)
    bbox = df.values[:, 0:3].max(axis=0) - df.values[:, 0:3].min(axis=0)

    positions = df.values[:, 0:3]
    colors = df.values[:, 3:6]
    normals = df.values[:, 6:9]
    uvs = df.values[:, 9:11]

    n_vertices = len(positions)
    print("no. of vertices: %d, no. of triangles: %d" % 
          (n_vertices, n_vertices//3))
    print("Centroid:", centroid)
    print("BBox:", bbox)

def main():  
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(win_w, win_h)
    glutCreateWindow(b"Textured Lit Bunny with VBO")
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse_func)
    glutPassiveMotionFunc(motion_func)
    glutMotionFunc(motion_func)
    glutIdleFunc(idle)
    init_gl_and_model()
    init_shaders()
    glutMainLoop()

if __name__ == "__main__":
    main()

#136 - abs(dot(N, L)) -> ทำให้แสงเข้าเข้าด้านในขนาดด้วย
#150 - ต้องทำ normalize อีกรอบ เพราะบางจุดมันไม่เป็นหนึ่งหน่วยแล้ว
#เมื่อต้องการสับเปลี่ยนการแสดงผลระหว่าง ground - phong ให้เปลี่ยนที่ vao (มี vao 2 ตัว)