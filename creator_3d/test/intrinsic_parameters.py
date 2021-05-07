from creator_3d.reconstuctor import camera_calibration
import numpy as np
import os

path_to_data = "C:\\Users\\hp\\Desktop\\3DCreator\\creator_3d\\data"

folder_name = "rock_head"
image_name = "1.jpg"
rock_head_image = os.path.join(path_to_data, folder_name, image_name)

calibrator = camera_calibration.Calibrator()

ROCK_HEAD_K = calibrator.get_intrinsic_matrix(f_mm=4.9,
                                              image_path=rock_head_image,
                                              sensor_width=6.08,
                                              sensor_height=4.56)

PPM_K = np.array([[2780.1700000000000728, 0, 1539.25],
                  [0, 2773.5399999999999636, 1001.2699999999999818],
                  [0, 0, 1]])

FOUNTAIN_K = np.array([[2759.48, 0, 1520.69],
                       [0, 2764.16, 1006.81],
                       [0, 0, 1]])

