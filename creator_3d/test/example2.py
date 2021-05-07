import cv2 as cv
import os
import numpy as np

from creator_3d.reconstuctor import camera_calibration

from glob import glob

#  ######################## Path Variables ##################################################
curr_dir_path = os.getcwd()

images_dir = "C:\\Users\\hp\\Desktop\\3DCreator\\creator_3d\\data\\rock"
calibration_image = f"{images_dir}\\img_0001.jpg"

img_pattern = images_dir + "\\img_????.jpg"  # './data/rdimage.???.ppm'
# images_dir = curr_dir_path + '/data/images/observatory'
# calibration_file_dir = curr_dir_path + '/data/calibration'


###########################################################################################

# def get_camera_intrinsic_params():
#     K = []
#     with open(calibration_file_dir + '/cameras.txt') as f:
#         lines = f.readlines()
#         calib_info = [float(val) for val in lines[0].split(' ')]
#         row1 = [calib_info[0], calib_info[1], calib_info[2]]
#         row2 = [calib_info[3], calib_info[4], calib_info[5]]
#         row3 = [calib_info[6], calib_info[7], calib_info[8]]
#
#         K.append(row1)
#         K.append(row2)
#         K.append(row3)
#
#     return K
#
#
# def get_pinhole_intrinsic_params():
#     K = []
#     with open(calibration_file_dir + '/camera_observatory.txt') as f:
#         lines = f.readlines()
#         calib_info = [float(val) for val in lines[0].split(' ')]
#         row1 = [calib_info[0], 0, calib_info[2]]
#         row2 = [0, calib_info[1], calib_info[3]]
#         row3 = [0, 0, 1]
#
#         K.append(row1)
#         K.append(row2)
#         K.append(row3)
#
#     return K


def rep_error_fn(opt_variables, points_2d, num_pts):
    P = opt_variables[0:12].reshape(3, 4)
    point_3d = opt_variables[12:].reshape((num_pts, 4))

    rep_error = []

    for idx, pt_3d in enumerate(point_3d):
        pt_2d = np.array([points_2d[0][idx], points_2d[1][idx]])

        reprojected_pt = np.matmul(P, pt_3d)
        reprojected_pt /= reprojected_pt[2]

        print("Reprojection Error \n" + str(pt_2d - reprojected_pt[0:2]))
        rep_error.append(pt_2d - reprojected_pt[0:2])


def write_simple_obj(mesh_v, mesh_f, filepath, verbose=False):
    """
    Saves 3d points which can be read in meshlab
    """
    with open(filepath, 'w') as fp:
        for v in mesh_v:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        if mesh_f is not None:
            for f in mesh_f+1:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    if verbose:
        print('mesh saved to: ', filepath)


def safe_mkdir(file_dir):
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)


def load_images(img_pattern, downscale=1):
    """
    Loads a set of images to self.imgs list
    """
    img_paths = sorted(glob(img_pattern))
    imgs = []
    for idx, this_path in enumerate(img_paths):
        try:
            this_img = cv.imread(this_path)
            if downscale > 1:
                this_img = cv.resize(this_img, (0, 0),
                                     fx=1/float(downscale),
                                     fy=1/float(downscale),
                                     interpolation=cv.INTER_LINEAR)
        except Exception as e:
            print("error loading img: %s" % (this_path))
        if this_img is not None:
            imgs.append(this_img)
            print("loaded img %d size=(%d,%d): %s" %
                  (idx, this_img.shape[0], this_img.shape[1], this_path))
    print("loaded %d images" % (len(imgs)))
    return imgs


if __name__ == "__main__":
    # Variables
    iter_num = 0
    prev_img = None
    prev_kp = None
    prev_desc = None

    calibrator = camera_calibration.Calibrator()
    K = intrinsic = calibrator.get_intrinsic_matrix(f_mm=35.0,
                                                    image_path=calibration_image,
                                                    sensor_width=23.6,
                                                    sensor_height=15.8)
    # np.array(get_pinhole_intrinsic_params(), dtype=np.float)

    R_t_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    R_t_1 = np.empty((3, 4))
    P1 = np.matmul(K, R_t_0)
    P2 = np.empty((3, 4))
    pts_4d = []
    X = np.array([])
    Y = np.array([])
    Z = np.array([])

    imgs = load_images(img_pattern=img_pattern)
    for img in imgs:

        # file = os.path.join(images_dir, filename)
        # ext = os.path.splitext(filename)[1]
        # if ext != ".jpg":
        #    continue

        # print("FILE_PATH: ", file)
        # img = cv.imread(file, 0)

        resized_img = img
        sift = cv.SIFT_create()
        kp, desc = sift.detectAndCompute(resized_img, None)

        if iter_num == 0:
            prev_img = resized_img
            prev_kp = kp
            prev_desc = desc
        else:
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=100)
            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(prev_desc, desc, k=2)
            good = []
            pts1 = []
            pts2 = []
            # ratio test as per Lowe's paper
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    good.append(m)
                    pts1.append(prev_kp[m.queryIdx].pt)
                    pts2.append(kp[m.trainIdx].pt)

            pts1 = np.array(pts1)
            pts2 = np.array(pts2)
            F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)
            print("The fundamental matrix \n" + str(F))

            # We select only inlier points
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]

            # draw_epipolar_lines(pts1, pts2, prev_img, resized_img)

            E = np.matmul(np.matmul(np.transpose(K), F), K)

            print("The new essential matrix is \n" + str(E))

            retval, R, t, mask = cv.recoverPose(E, pts1, pts2, K)

            print("I+0 \n" + str(R_t_0))

            print("Mullllllllllllll \n" + str(np.matmul(R, R_t_0[:3, :3])))

            R_t_1[:3, :3] = np.matmul(R, R_t_0[:3, :3])
            R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3, :3], t.ravel())

            print("The R_t_0 \n" + str(R_t_0))
            print("The R_t_1 \n" + str(R_t_1))

            P2 = np.matmul(K, R_t_1)

            print("The projection matrix 1 \n" + str(P1))
            print("The projection matrix 2 \n" + str(P2))

            pts1 = np.transpose(pts1)
            pts2 = np.transpose(pts2)

            print("Shape pts 1\n" + str(pts1.shape))

            points_3d = cv.triangulatePoints(P1, P2, pts1, pts2)
            points_3d /= points_3d[3]

            # P2, points_3D = bundle_adjustment(points_3d, pts2, resized_img, P2)
            opt_variables = np.hstack((P2.ravel(), points_3d.ravel(order="F")))
            num_points = len(pts2[0])
            rep_error_fn(opt_variables, pts2, num_points)

            X = np.concatenate((X, points_3d[0]))
            Y = np.concatenate((Y, points_3d[1]))
            Z = np.concatenate((Z, points_3d[2]))

            R_t_0 = np.copy(R_t_1)
            P1 = np.copy(P2)
            prev_img = resized_img
            prev_kp = kp
            prev_desc = desc

        iter_num = iter_num + 1

    pts_4d.append(X)
    pts_4d.append(Y)
    pts_4d.append(Z)

    output_dir = './tmp'
    safe_mkdir(output_dir)

    write_simple_obj(np.array(pts_4d).T, None, filepath=os.path.join(output_dir, 'output.obj'))

    # viz_3d(np.array(pts_4d))
