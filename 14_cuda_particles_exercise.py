import sys
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from numpy import zeros, array, dtype, int32, float32
from numpy.linalg import norm
from math import tan, pi, pow
from random import uniform, seed
from ctypes import c_void_p
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

win_w, win_h, pause, box_rotate = 1024, 768, True, 0
n_particles = 163840//100
delta_time, global_damping = 0.5, 1.0
particle_radius, boundary_damping = 1/24, -0.8
spring, damping, shear, attraction = 0.5, 0.01, 0.1, 0.0
gravity = array((0.0, -0.0005, 0.0), dtype=float32)
particle_vao = None

cuda_code = '''
#include <stdio.h>
inline __host__ __device__ float3 make_float3(float s)
{
    return make_float3(s, s, s);
}
inline __host__ __device__ float4 make_float4(float3 a, float w)
{
    return make_float4(a.x, a.y, a.z, w);
}
inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ float3 operator*(float b, float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ void operator*=(float3 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}
inline __host__ __device__ float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float length(float3 v)
{
    return sqrtf(dot(v, v));
}

extern "C" {
__global__ void compute_posvel(float4* __restrict__ position,
    float4* __restrict__ velocity, float4* __restrict__ force, int n_particles,
    float delta_time, float *gravity, float global_damping,
    float particle_radius, float boundary_damping)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;;
	if (idx >= n_particles)
		return;

    float4 posData = position[idx];
    float4 velData = velocity[idx];
    float4 frcData = force[idx];
    float3 pos = make_float3(posData.x, posData.y, posData.z);
    float3 vel = make_float3(velData.x, velData.y, velData.z);
    float3 frc = make_float3(frcData.x, frcData.y, frcData.z);

    vel += (make_float3(gravity[0], gravity[1], gravity[2]) + frc) * delta_time;
    vel *= global_damping;

    // new position = old position + velocity * deltaTime
    pos += vel * delta_time;

    if (pos.x > 1.0f - particle_radius) {
      	pos.x = 1.0f - particle_radius;
      	vel.x *= boundary_damping;
    }

    if (pos.x < -1.0f + particle_radius) {
      	pos.x = -1.0f + particle_radius;
      	vel.x *= boundary_damping;
    }

    if (pos.y > 1.0f - particle_radius) {
      	pos.y = 1.0f - particle_radius;
      	vel.y *= boundary_damping;
    }

    if (pos.y < -1.0f + particle_radius) {
      	pos.y = -1.0f + particle_radius;
      	vel.y *= boundary_damping;
    }

    if (pos.z > 1.0f - particle_radius) {
      	pos.z = 1.0f - particle_radius;
      	vel.z *= boundary_damping;
    }

    if (pos.z < -1.0f + particle_radius) {
      	pos.z = -1.0f + particle_radius;
      	vel.z *= boundary_damping;
    }

    position[idx] = make_float4(pos, posData.w);
    velocity[idx] = make_float4(vel, velData.w);
}

__global__ void collide(float4* __restrict__ position,
    float4* __restrict__ velocity, int n_particles,
    float particle_radius, float spring, float damping, 
    float shear, float attraction, 
    float* __restrict__ force)
{   
    int idx = blockIdx.x * blockDim.x + threadIdx.x;;
    if (idx >= n_particles*n_particles)
        return;

    int A = idx / n_particles;
    int B = idx % n_particles;
    if (A == B)
        return;

    float4 posDataA = position[A];
    float4 posDataB = position[B];
    float4 velDataA = velocity[A];
    float4 velDataB = velocity[B];
    float3 posA = make_float3(posDataA.x, posDataA.y, posDataA.z);
    float3 posB = make_float3(posDataB.x, posDataB.y, posDataB.z);
    float3 velA = make_float3(velDataA.x, velDataA.y, velDataA.z);
    float3 velB = make_float3(velDataB.x, velDataB.y, velDataB.z);
    float3 relPos = posB - posA;
    float dist = length(relPos);
    float collideDist = 2 * particle_radius;

    float3 f = make_float3(0.0f);

    if (dist < collideDist) {
        ;
    }
    atomicAdd(force + 4 * A,   f.x);
    atomicAdd(force + 4 * A+1, f.y);
    atomicAdd(force + 4 * A+2, f.z);
}

}
    '''

cuda_mod = SourceModule(cuda_code, no_extern_c=True)

def compute_posvel(n_threads_per_block=512, verbose=False):
    start = cuda.Event()
    end   = cuda.Event()
    start.record()

    one_more_block = n_particles % n_threads_per_block
    grid = (n_particles // n_threads_per_block + one_more_block, 1, 1)
    block = (n_threads_per_block, 1, 1)

    d_n_particles = int32(n_particles)
    d_delta_time = float32(delta_time)
    d_global_damping = float32(global_damping)
    d_particle_radius = float32(particle_radius)
    d_boundary_damping = float32(boundary_damping)

    cuda_func = cuda_mod.get_function("compute_posvel")
    cuda_func(d_position, d_velocity, d_force, d_n_particles, 
        d_delta_time, cuda.In(gravity), d_global_damping, 
        d_particle_radius, d_boundary_damping, grid=grid, block=block)

    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    if verbose:
        print("compute_posvel: CUDA clock timing: %.4f secs" % secs)

def collide(n_threads_per_block=512, verbose=False):
    start = cuda.Event()
    end   = cuda.Event()
    start.record()

    one_more_block = (n_particles*n_particles) % n_threads_per_block
    grid = (n_particles*n_particles // n_threads_per_block + one_more_block, 1, 1)
    block = (n_threads_per_block, 1, 1)

    d_n_particles = int32(n_particles)
    d_particle_radius = float32(particle_radius)
    d_spring = float32(spring)
    d_damping = float32(damping)
    d_shear = float32(shear)
    d_attraction = float32(attraction)

    cuda_func = cuda_mod.get_function("collide")
    cuda_func(d_position, d_velocity, d_n_particles, 
        d_particle_radius, d_spring, d_damping, d_shear, d_attraction,
        d_force, grid=grid, block=block)

    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    if verbose:
        print("collide: CUDA clock timing: %.4f secs" % secs)

def print_shader_info_log(shader, prompt=""):
    result = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not result:
        print("%s: %s" % (prompt, glGetShaderInfoLog(shader).decode("utf-8")))
        return -1
    else:
        return 0

def print_program_info_log(program, prompt=""):
    result = glGetProgramiv(program, GL_LINK_STATUS)
    if not result:
        print("%s: %s" % (prompt, glGetProgramInfoLog(program).decode("utf-8")))
        return -1
    else:
        return 0

def compile_program(vertex_code, fragment_code):
    vert_id = glCreateShader(GL_VERTEX_SHADER)
    frag_id = glCreateShader(GL_FRAGMENT_SHADER)

    glShaderSource(vert_id, vertex_code)
    glShaderSource(frag_id, fragment_code)

    glCompileShader(vert_id)
    glCompileShader(frag_id)

    print_shader_info_log(vert_id, "Vertex Shader")
    print_shader_info_log(frag_id, "Fragment Shader")

    prog_id = glCreateProgram()
    glAttachShader(prog_id, vert_id)
    glAttachShader(prog_id, frag_id)

    glLinkProgram(prog_id)
    print_program_info_log(prog_id, "Link error")

    return prog_id

def create_shaders():
    global prog_id

    vertex_code = '''
#version 120
uniform float point_radius;  // point size in world space
uniform float point_scale;   // scale to calculate size in pixels
void main()
{                            
    // calculate window-space point size
    vec3 posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
    float dist = length(posEye);
    gl_PointSize = point_radius * (point_scale / dist);

    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);

    gl_FrontColor = gl_Color;
}
        '''
    fragment_code = '''
#version 120
void main() 
{
    const vec3 lightDir = vec3(0.577, 0.577, 0.577);

    // calculate normal from texture coordinates
    vec3 N;
    N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(N.xy, N.xy);

    if (mag > 1.0) discard;   // kill pixels outside circle

    N.z = sqrt(1.0-mag);

    // calculate lighting
    float diffuse = max(0.0, dot(lightDir, N));

    gl_FragColor = gl_Color * diffuse;
}
        '''
    try:
        prog_id = compile_program(vertex_code, fragment_code)
    except:
        sys.exit(1)        
    if prog_id:
        glUseProgram(prog_id)
        glUniform1f(glGetUniformLocation(prog_id, "point_scale"), 
            win_h / tan(60*0.5*pi/180.0))
        glUniform1f(glGetUniformLocation(prog_id, "point_radius"), particle_radius)

def idle():
    global box_rotate

    compute_posvel(verbose=False)
    cuda.memcpy_htod(d_force, h_force)
    collide(verbose=False)
    cuda.memcpy_dtoh(h_position, d_position)
    box_rotate += 0.1
    glutPostRedisplay()

def keyboard(ikey, x, y):
    global pause

    key = ikey.decode("utf-8")
    if key == ' ':
        pause = not pause
        glutIdleFunc(None if pause else idle)
    elif key == '/':
        jitter = particle_radius * 0.01
        spacing = particle_radius * 2
        s = int(pow(n_particles, 1/3)+1)
        for z in range(s):
            for y in range(s):
                for x in range(s):
                    i = (z*s*s) + (y*s) + x
                    if i < n_particles:
                        h_position[i][0] = (spacing * x) + particle_radius - 1.0 + uniform(-1, 1)*jitter
                        h_position[i][1] = -0.5 + (spacing * y) + particle_radius - 0.2 + uniform(-1, 1)*jitter
                        h_position[i][2] = (spacing * z) + particle_radius - 1.0 + uniform(-1, 1)*jitter
                        h_position[i][3] = 1.0
        cuda.memcpy_htod(d_position, h_position)
    elif key.lower() == 'q':
        exit(0)
    glutPostRedisplay()

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(1.8, 0, 2, 0, -0.2, 0, 0, 1, 0)
    glRotatef(box_rotate, 0, 1, 0)
    
    glUseProgram(0)
    glColor3f(1.0, 1.0, 1.0)
    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glutWireCube(2.0)

    glUseProgram(prog_id)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(4, GL_FLOAT, 0, h_position)
    glDrawArrays(GL_POINTS, 0, n_particles)
    glutSwapBuffers()

def reshape(w, h):
    global win_w, win_h

    win_w, win_h = w, h
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., w/h, 0.1, 100)

colors = array((( 1.0, 0.0, 0.0, 0 ),
                ( 1.0, 0.5, 0.0, 0 ),
                ( 1.0, 1.0, 0.0, 0 ),
                ( 0.0, 1.0, 0.0, 0 ),
                ( 0.0, 1.0, 1.0, 0 ),
                ( 0.0, 0.0, 1.0, 0 ),
                ( 1.0, 0.0, 1.0, 0 )), dtype=float32)

def colorRamp(t):
    ncolors = len(colors)
    t = t * (ncolors-1)
    i = int(t)
    u = t - i
    return colors[i] + u * (colors[i+1]-colors[i])

def init_particles():
    global h_position, h_velocity, h_color, h_force, d_position, d_velocity, d_force

    seed(12345)
    h_position = zeros((n_particles, 4), dtype=float32)
    h_velocity = zeros((n_particles, 4), dtype=float32)
    h_color = zeros((n_particles, 4), dtype=float32)
    h_force = zeros((n_particles, 4), dtype=float32)
    for i in range(n_particles):
        h_color[i] = colorRamp(i/n_particles)

    d_position = cuda.mem_alloc(h_position.size * h_position.dtype.itemsize)
    d_velocity = cuda.mem_alloc(h_velocity.size * h_velocity.dtype.itemsize)
    d_force = cuda.mem_alloc(h_force.size * h_force.dtype.itemsize)
    cuda.memcpy_htod(d_position, h_position)
    cuda.memcpy_htod(d_velocity, h_velocity)
    cuda.memcpy_htod(d_force, h_force)
    keyboard(b'/', 0, 0)

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(4, GL_FLOAT, 0, h_position)
    glColorPointer(4, GL_FLOAT, 0, h_color)

def init_gl(clear_color=(0, 0, 0, 0)):
    glClearColor(*clear_color)
    glEnable(GL_POINT_SPRITE)
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)
    glDepthMask(GL_TRUE)
    glEnable(GL_DEPTH_TEST)

    lists = [["Vendor", GL_VENDOR], ["Renderer",GL_RENDERER],
             ["OpenGL Version", GL_VERSION], 
             ["GLSL Version", GL_SHADING_LANGUAGE_VERSION]]
    for x in lists:
        print("%s: %s" % (x[0], glGetString(x[1]).decode("utf-8")))

def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(win_w, win_h)
    glutInitWindowPosition(100, 50)
    glutCreateWindow(b"CUDA Particles")
    init_gl(clear_color=(0.25, 0.25, 0.25, 1.0))
    glutKeyboardFunc(keyboard)
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    create_shaders()
    init_particles()
    glutMainLoop()

if __name__ == "__main__":
    main()    