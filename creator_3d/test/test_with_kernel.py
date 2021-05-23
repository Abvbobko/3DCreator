"""This test uses math ideas from computer vision course
    and try to be like a final code
"""

import os

from creator_3d.reconstuctor.actions.steps import geometric_verification, feature_matching, feature_extraction
from creator_3d.reconstuctor import camera_calibration

import cv2


def get_file_list(data_path, ext, num_of_files=-1):
    file_list = []
    cnt = 0
    for file in os.listdir(DATA_PATH):
        if cnt == num_of_files:
            break
        if file.split(".")[-1] == extension:
            file_list.append(file)
    return file_list


DATA_PATH = "C:\\Users\\hp\\Desktop\\3DCreator\\creator_3d\\data\\rock_head\\"
extension = 'jpg'

sift = feature_extraction.SIFT()
matcher = feature_matching.BFMatcher()
geometric_verifier = geometric_verification.PointForFMatrix()

image_files_list = get_file_list(DATA_PATH, extension)
num_of_image_files = len(image_files_list)

calibrator = camera_calibration.Calibrator()
# todo: для теста ввести тут нормальные параметры
image_1_path = os.path.join(DATA_PATH, image_files_list[0])
k = calibrator.run(image_1_path=image_1_path)
print(k)

# todo: передавать следующей картинке координаты текущей относительно первой
# todo: то есть чтобы все было относительно первой картинки в итоге

for i in range(num_of_image_files - 1):
    image_1_path = os.path.join(DATA_PATH, image_files_list[i])
    image_2_path = os.path.join(DATA_PATH, image_files_list[i+1])

    image_1 = cv2.imread(image_1_path)
    image_2 = cv2.imread(image_2_path)

    key_points_1, descriptors_1 = sift.run(image=image_1)
    key_points_2, descriptors_2 = sift.run(image=image_2)

    good = matcher.match_features(key_points_1, descriptors_1, key_points_2, descriptors_2)

    pts1, pts2, F = geometric_verifier.verify(good, image_1, image_2, key_points_1, key_points_2)


