from numpy import array, ndarray, zeros, dot, cross, float32, identity
from numpy.linalg import norm
from math import sqrt, sin, cos, tan, acos, pi

def Identity():
    return array(((1, 0, 0, 0),
                  (0, 1, 0, 0),
                  (0, 0, 1, 0),
                  (0, 0, 0, 1)), dtype=float32)

def normalize(v):
    l = norm(v)
    if l == 0:
        return v
    else:
        return v/l

def Translate(tx, ty, tz):
    return array(((1, 0, 0, tx),
                  (0, 1, 0, ty),
                  (0, 0, 1, tz),
                  (0, 0, 0, 1)), dtype=float32)

def Scale(sx, sy, sz):
    return array(((sx, 0, 0, 0),
                  (0, sy, 0, 0),
                  (0, 0, sz, 0),
                  (0, 0, 0, 1)), dtype=float32)

def Rotate(angle, x, y, z, radian=False):
    sqr_a = x * x
    sqr_b = y * y
    sqr_c = z * z
    len2  = sqr_a + sqr_b + sqr_c

    if not radian:
       angle = angle/180*pi
    k2    = cos(angle)
    k1    = (1.0-k2) / len2
    k3    = sin(angle) / sqrt(len2)
    k1ab  = k1 * x * y
    k1ac  = k1 * x * z
    k1bc  = k1 * y * z
    k3a   = k3 * x
    k3b   = k3 * y
    k3c   = k3 * z

    return array(((k1*sqr_a+k2, k1ab-k3c, k1ac+k3b, 0.0),
                   (k1ab+k3c, k1*sqr_b+k2, k1bc-k3a, 0.0),
                   (k1ac-k3b, k1bc+k3a, k1*sqr_c+k2, 0.0),
                   (0.0, 0.0, 0.0, 1.0)), dtype=float32)

def LookAt(eyex, eyey, eyez, atx, aty, atz, upx, upy, upz):
    eye = array((eyex, eyey, eyez), dtype=float32)
    at = array((atx, aty, atz), dtype=float32)
    up = array((upx, upy, upz), dtype=float32)
    Z = normalize(eye - at)
    Y = normalize(up)
    X = normalize(cross(Y, Z))
    Y = normalize(cross(Z, X))

    return array(((X[0], X[1], X[2], -dot(X, eye)),
                  (Y[0], Y[1], Y[2], -dot(Y, eye)),
                  (Z[0], Z[1], Z[2], -dot(Z, eye)),
                  (0, 0, 0, 1)), dtype=float32)

def cot(n):
    return 1/tan(n)

def Perspective(fovy, aspect, zNear, zFar):
    cot_val = cot((fovy*pi/180/2))
    return array(((cot_val/aspect, 0, 0, 0),
                  (0, cot_val, 0, 0),
                  (0, 0, -(zFar+zNear)/(zFar-zNear), -2*zNear*zFar/(zFar-zNear)),
                  (0, 0, -1, 0)), dtype=float32)
