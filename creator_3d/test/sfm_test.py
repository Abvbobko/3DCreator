import matplotlib.pyplot as plt
import numpy as np
from math import pi, sin, cos, sqrt
import cv2


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


def generate_points(L=300):
    I = np.zeros((L, L))
    f = L
    u0 = L / 2
    v0 = L / 2
    Mint = np.array([[f, 0, u0],
                     [0, f, v0],
                     [0, 0, 1]])
    DEGREE_TO_RADIAN = pi / 180
    p_m = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2],
                    [2, 1, 0, 2, 1, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, -1, -1, -1, -2, -2, -2, 0, 0, -1, -1, -2, -2],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    # npts = len(p)
    ax = 120 * DEGREE_TO_RADIAN
    ay = 0 * DEGREE_TO_RADIAN
    az = 60 * DEGREE_TO_RADIAN
    Rx = np.array([[1, 0, 0],
                   [0, cos(ax), -sin(ax)],
                   [0, sin(ax), cos(ax)]])
    Ry = np.array([[cos(ay), 0, sin(ay)],
                   [0, 1, 0],
                   [-sin(ay), 0, cos(ay)]])
    Rz = np.array([[cos(az), -sin(az), 0],
                   [sin(az), cos(az), 0],
                   [0, 0, 1]])
    R_m_c1 = Rx.dot(Ry).dot(Rz)
    Pmorg_c1 = np.array([[0],
                         [0],
                         [5]])
    M = np.concatenate((R_m_c1, Pmorg_c1), axis=1)

    p1 = M.dot(p_m)
    p1[0, :] = p1[0, :] / p1[2, :]
    p1[1, :] = p1[1, :] / p1[2, :]
    p1[2, :] = p1[2, :] / p1[2, :]
    u1 = Mint.dot(p1)
    # draw_2d_points(u)
    # generate points 2
    ax = 0 * DEGREE_TO_RADIAN
    ay = -25 * DEGREE_TO_RADIAN
    az = 0
    Rx = np.array([[1, 0, 0],
                   [0, cos(ax), -sin(ax)],
                   [0, sin(ax), cos(ax)]])
    Ry = np.array([[cos(ay), 0, sin(ay)],
                   [0, 1, 0],
                   [-sin(ay), 0, cos(ay)]])
    Rz = np.array([[cos(az), -sin(az), 0],
                   [sin(az), cos(az), 0],
                   [0, 0, 1]])
    R_c2_c1 = Rx.dot(Ry).dot(Rz)
    Pc2org_c1 = np.array([[3],
                          [0],
                          [1]])

    H_m_c1 = np.concatenate((np.concatenate((R_m_c1,
                                             Pmorg_c1), axis=1),
                             np.array([[0, 0, 0, 1]])), axis=0)

    H_c2_c1 = np.concatenate((np.concatenate((R_c2_c1,
                                              Pc2org_c1), axis=1),
                              np.array([[0, 0, 0, 1]])), axis=0)

    H_c1_c2 = np.linalg.inv(H_c2_c1)
    H_m_c2 = H_c1_c2.dot(H_m_c1)

    R_m_c2 = H_m_c2[0:3, 0:3]
    Pmorg_c2 = H_m_c2[0:3, 3]
    Pmorg_c2 = np.array([[Pmorg_c2[0]],
                         [Pmorg_c2[1]],
                         [Pmorg_c2[2]]])

    M = np.concatenate((R_m_c2, Pmorg_c2), axis=1)
    p2 = M.dot(p_m)
    p2[0, :] = p2[0, :] / p2[2, :]
    p2[1, :] = p2[1, :] / p2[2, :]
    p2[2, :] = p2[2, :] / p2[2, :]
    u2 = Mint.dot(p2)

    # draw_2d_points(u)
    return u1, u2


def scale(points):
    # scale and translate points so that centroid of the points is at the origin
    # and the average distance of the points of the origin is equal to sqrt(2)

    # xn is 2xN matrix
    xn = points[0:2, :]  # x and y of p_1
    N = xn.shape[1]

    # this is the (x, y) centroid of the points
    # вектор столбец (сумма всех элементов каждой строки) поделить на количество
    t = (1 / N) * xn.sum(axis=1).reshape(2, 1)

    # center the points; xnc is a 2xN matrix
    # вычесть от каждой строки строку, где каждый элемент t (среднее значение в строке)
    xnc = xn - t.dot(np.ones((1, N)))

    # dist of each new point to 0, 0; dc is 1xN vector
    # сложить x^2 + y^2 для каждой точки и найти корень (расстояние точки до 0)
    dc = np.sqrt(np.square(xnc).sum(axis=0))

    d_avg = (1 / N) * dc.sum()  # average distance to the origin
    s = sqrt(2) / d_avg  # the scale factor, so that avg dist is sqrt(2)
    print("t: ", t)
    T = np.array([[s, 0, -s * t[0][0]],
                  [0, s, -s * t[1][0]],
                  [0, 0, 1]])

    points_scaled = T.dot(points)
    return points_scaled, T


def precondition(points_1, points_2):
    # precondition
    p_1s, T1 = scale(points_1)
    p_2s, T2 = scale(points_2)
    return p_1s, p_2s, T1, T2


def compute_essential_matrix(points_0, points_1):
    """Solve
        Ax = 0
        where A is (x0x1 x0y1 x0 y0x1 y0y1 y0 x1 y1 1),
            where x0 - points_0 x
                  y0 - points_0 y
                  x1 - points_1 x
                  y1 - points_1 y
        and x is (E11 E12 E13 E21 E22 E23 E31 E32 E33)^T
    Returns:
          E - essential matrix
    """
    print("Shape p0: ", points_0.shape)
    x0 = points_0[0, :]
    y0 = points_0[1, :]
    x1 = points_1[0, :]
    y1 = points_1[1, :]

    A = np.array([x0 * x1,
                  x0 * y1,
                  x0,
                  y0 * x1,
                  y0 * y1,
                  y0,
                  x1,
                  y1,
                  np.ones(points_0.shape[1])]).T

    U, D, V = np.linalg.svd(A)
    E = V[:, -1]  # get last column of V

    E_scale = E.reshape((3, 3)).T
    return E_scale


def postcondition(E_scaled):
    U, D, V = np.linalg.svd(E_scaled)
    E_scaled = U.dot(np.diag([1, 1, 0])).dot(V.T)
    return E_scaled


def undo_scaling(E_scaled, T1, T2):
    return T1.T.dot(E_scaled).dot(T2)


def to_skew_matrix(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def reconstruct(p1, p2, E):
    def create_camera_position_matrix(u, w, v, add_col):
        """add_col - additional column"""
        return np.concatenate(
            (
                np.concatenate((u.dot(w).dot(v), add_col.reshape((3, 1))), axis=1),
                np.array([[0, 0, 0, 1]])
            )
        )

    U, D, V = np.linalg.svd(E)
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    Hresult_c2_c1 = []
    # тут 4 возможных положения камеры, но подойдет только 1
    Hresult_c2_c1.append(create_camera_position_matrix(U, W, V.T, U[:, 2]))
    Hresult_c2_c1.append(create_camera_position_matrix(U, W, V.T, -U[:, 2]))
    Hresult_c2_c1.append(create_camera_position_matrix(U, W.T, V.T, U[:, 2]))
    Hresult_c2_c1.append(create_camera_position_matrix(U, W.T, V.T, -U[:, 2]))
    # make sure that rotation component is a legal rotation matrix
    # rotation matrices is right handed
    for k in range(0, 4):
        if np.linalg.det(Hresult_c2_c1[k][0:3, 0:3]) < 0:
            Hresult_c2_c1[k][0:3, 0:3] = -Hresult_c2_c1[k][0:3, 0:3]

    p1x = to_skew_matrix(p1[:, 0])
    p2x = to_skew_matrix(p2[:, 0])
    M1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])

    Hest_c2_c1 = None
    for i in range(0, 4):
        Hresult_c1_c2 = np.linalg.inv(Hresult_c2_c1[i])
        M2 = Hresult_c1_c2[0:3, 0:4]
        A = np.concatenate((p1x.dot(M1),
                            p2x.dot(M2)))

        U, D, V = np.linalg.svd(A)
        P = V[:, -1]
        P1est = P / P[-1]

        P2est = Hresult_c1_c2.dot(P1est)

        if P1est[2] > 0 and P2est[2] > 0:
            Hest_c2_c1 = Hresult_c2_c1[i]
            break

    if Hest_c2_c1 is None:
        return

    # reconstruct rest points
    Hest_c2_c1 = np.linalg.inv(Hest_c2_c1)
    M2est = Hest_c2_c1[0:3, :]

    P_result = []
    for i in range(len(p1[0])):
        p1x = to_skew_matrix(p1[:, i])
        p2x = to_skew_matrix(p2[:, i])
        A = np.concatenate((p1x.dot(M1),
                            p2x.dot(M2est)))

        U, D, V = np.linalg.svd(A)
        P = V[:, -1]
        P_result.append(P / P[-1])

    return P_result




# size of image in pixels
L = 300


generate_points(L)
# define f, u0, v0
f = L
u0 = L / 2
v0 = L / 2

# initial intrinsic camera matrix (K)
K = np.array([[f, 0, u0],
              [0, f, v0],
              [0, 0, 1]])

# points of the cube

u1, u2 = generate_points()

# draw_2d_points(u1)
# draw_2d_points(u2)

# normalize image points
p_1 = np.linalg.inv(K).dot(u1)
p_2 = np.linalg.inv(K).dot(u2)

p_1, p_2, T1, T2 = precondition(p_1, p_2)
E = compute_essential_matrix(p_1, p_2)

E = postcondition(E)
E = undo_scaling(E, T1, T2)
# ---------------------------------------------
print(E)

# reconstruction part
p = reconstruct(p_1, p_2, E)

draw_3d_points(np.array(p).T)

a = np.array([[16, 8, 8], [16, 16, 16]])
b = np.array([4, 4, 4])
c = np.array([[0, 0, 0, 0]])

# todo: E не сходится - перепроверить все расчеты и числа!!!!!!!!

# todo: потом можно попробовать юзать это
# E, mask = cv2.findEssentialMat(u1, u2, K, cv2.RANSAC, prob=0.999, threshold=1.0)


#       0 -1       0
# -0.3615  0 -3.1415
#       0  3       0
