import matplotlib.pyplot as plt
import numpy as np
from math import pi, sin, cos, sqrt


def draw_3d_points(points):
    # draw cube
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(points[0], points[1], points[2], 'b.')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    # ax.view_init(elev=135, azim=90)
    plt.show()


def draw_2d_points(points):
    # draw cube
    fig = plt.figure()
    ax = fig.gca()
    fig.gca().invert_yaxis()
    ax.plot(points[0], points[1], 'b.')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    plt.show()


def scale(points):
    # scale and translate points so that centroid of the points is at the origin
    # and the average distance of the points of the origin is equal to sqrt(2)

    # xn is 2xN matrix
    xn = points[0:2][:]  # x and y of p_1
    N = xn.shape[1]

    # this is the (x, y) centroid of the points
    # вектор столбец (сумма всех элементов каждой строки) поделить на количество
    t = (1 / N) * xn.sum(axis=1).reshape(2, 1)

    # center the points; xnc is a 2xN matrix
    # вычесть от каждой строки строку, где каждый элемент t (среднее значение в строке)
    xnc = xn - t.dot(np.ones((1, N)))

    # dist of each new point to 0, 0; dc is 1xN vector
    dc = np.sqrt(
        np.square(xnc).sum(axis=0))  # сложить x^2 + y^2 для каждой точки и найти корень (расстояние точки до 0)

    d_avg = (1 / N) * dc.sum()  # average distance to the origin
    s = 2 ** 2 / d_avg  # the scale factor, so that avg dist is sqrt(2)
    T1 = np.array([[s, 0, -s * t[0][0]],
                   [0, s, -s * t[1][0]],
                   [0, 0, 1]])

    points_scaled = T1.dot(points)
    return points_scaled


def precondition(points_1, points_2):
    # precondition
    p_1s = scale(points_1)
    p_2s = scale(points_2)
    return p_1s, p_2s


def compute_essential_matrix(x, y):
    """Solve
        Ax = 0
        where A is (x0x1 x0y1 x0 y0x1 y0y1 y0 x1 y1 1)
        and x is (E11 E12 E13 E21 E22 E23 E31 E32 E33)^T
    Returns:
          E - essential matrix
    """
    A = np.array([])

DEGREE_TO_RADIAN = pi / 180

# size of image in pixels
L = 300
I = np.zeros((L, L))

# define f, u0, v0
f = L
u0 = L/2
v0 = L/2


# initial intrinsic camera matrix (K)
K = np.array([[200, 0,   150],
             [0,   300, 150],
             [0,   0,   1]])

# points of the cube
p_m = np.array([[0, 0, 0,  0,  0,  0,  0,  0,  0, 1, 2,  1,  2,  1,  2],
                [2, 1, 0,  2,  1,  0,  2,  1,  0, 0, 0,  0,  0,  0,  0],
                [0, 0, 0, -1, -1, -1, -2, -2, -2, 0, 0, -1, -1, -2, -2],
                [1, 1, 1,  1,  1,  1,  1,  1,  1, 1, 1,  1,  1,  1,  1]])


u1 = np.array([[61.4195,  102.1798, 150.0000, 68.37690, 106.2098, 150.0000, 74.32090, 109.6134, 150.0000, 176.0870, 196.1538, 174.0000, 192.8571, 172.2222, 190.0000],
               [124.4290, 136.1955, 150.0000, 167.2490, 181.1490, 197.2377, 203.8325, 219.1146, 236.6025, 127.4080, 110.0296, 170.7846, 150.0000, 207.7350, 184.6410],
               [1,        1,        1,        1,        1,        1,        1,        1,        1,        1,        1,        1,        1,        1,        1]])

u2 = np.array([[45.52720, 63.45680, 86.94470, 61.66200, 80.26530, 104.1468, 75.79810, 94.75070, 118.6606, 135.5451, 176.3357, 147.5739, 184.5258, 157.8633, 191.6139],
               [126.5989, 136.7293, 150.0000, 165.9997, 180.2731, 198.5963, 200.5196, 217.7991, 239.5982, 125.7710, 105.4355, 172.3407, 150.0000, 212.1766, 188.5687],
               [1,        1,        1,        1,        1,        1,        1,        1,        1,        1,        1,        1,        1,        1,        1]])


# draw_2d_points(u1)
# draw_2d_points(u2)

# normalize image points
p_1 = np.linalg.inv(K).dot(u1)
p_2 = np.linalg.inv(K).dot(u2)

p_1 = scale(p_1)
p_2 = scale(p_2)









