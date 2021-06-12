from creator_3d.reconstuctor import camera_calibration
import numpy as np
import os
from creator_3d.data_access.data_controller import DataController

path_to_data = "C:\\Users\\hp\\Desktop\\3DCreator\\data"

folder_name = "rock_small"
image_name = "img_0001.jpg"
rock_head_image = os.path.join(path_to_data, folder_name, image_name)

calibrator = camera_calibration.Calibrator()

book_dir = "book_test"
book_image_name = "IMG_3426.JPG"
book_image_path = os.path.join(path_to_data, book_dir, book_image_name)
image_size = DataController.read_pil_image(book_image_path).size
CANON = camera_calibration.Camera(focal_length=33,
                                  sensor_width=22.3,
                                  sensor_height=14.9,
                                  image_width=image_size[0],
                                  image_height=image_size[1]).K

# ROCK_K = calibrator.get_intrinsic_matrix(f_mm=35,
#                                          image_path=rock_head_image,
#                                          sensor_width=23.6,
#                                          sensor_height=15.8)
#
# ROCK_HEAD_K = calibrator.get_intrinsic_matrix(f_mm=4.9,
#                                               image_path=rock_head_image,
#                                               sensor_width=6.08,
#                                               sensor_height=4.56)
#
# PPM_K = np.array([[2780.1700000000000728, 0, 1539.25],
#                   [0, 2773.5399999999999636, 1001.2699999999999818],
#                   [0, 0, 1]])
#
# FOUNTAIN_K = np.array([[2759.48, 0, 1520.69],
#                        [0, 2764.16, 1006.81],
#                        [0, 0, 1]])
#
# KERMIT_K = np.array([[182.363977486, 0, 640/2],
#                      [0, 243, 480/2],
#                      [0, 0, 1]])
# # KERMIT_K = calibrator.get_intrinsic_matrix(f_mm=5.4,
# #                                            image_path=rock_head_image,
# #                                            sensor_width=5.33,
# #                                            sensor_height=4)
#
# # 1/2.9" (~ 4.96 x 3.72 mm)
# banana_folder_name = 'banana'
# banana_image_name = '1.jpg'
# banana_image_path = os.path.join(path_to_data, banana_folder_name, banana_image_name)
#
# HUAWEI_P20_LITE_K = calibrator.get_intrinsic_matrix(f_mm=30,
#                                                     image_path=banana_image_path,
#                                                     sensor_width=4.96,
#                                                     sensor_height=3.72)


table_folder_name = 'ferrari/air'
table_image_name = 'IMAG0371.jpg'
table_image_path = os.path.join(path_to_data, table_folder_name, table_image_name)

# MOBIUS = calibrator.get_intrinsic_matrix(f_mm=2.5,
#                                          image_path=table_image_path,
#                                          sensor_width=5.07,
#                                          sensor_height=3.38)
