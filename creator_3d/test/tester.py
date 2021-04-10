"""This code just for run other code in test mode
"""

from creator_3d.reconstuctor.actions import feature_extraction
from creator_3d.reconstuctor.actions import feature_matching
from creator_3d.reconstuctor.actions import geometric_verification
from creator_3d.reconstuctor.actions import camera_calibration

import cv2
import matplotlib.pyplot as plt
import numpy as np


sift = feature_extraction.SIFT()
matcher = feature_matching.BFMatcher()
geometric_verifier = geometric_verification.PointForFMatrix()


data_path = "C:\\Users\\hp\\Desktop\\3DCreator\\creator_3d\\data\\rock_head\\"
image_1_path = data_path + "1.jpg"
image_2_path = data_path + "2.jpg"

image_1 = cv2.imread(image_1_path)
image_2 = cv2.imread(image_2_path)

key_points_1, descriptors_1 = sift.run(image=image_1)
key_points_2, descriptors_2 = sift.run(image=image_2)

good = matcher.match_features(key_points_1, descriptors_1, key_points_2, descriptors_2)

pts1, pts2, F = geometric_verifier.verify(good, image_1, image_2, key_points_1, key_points_2)

calibrator = camera_calibration.Calibrator()
k = calibrator.run(image_1_path=image_1_path)
print(k)

# # построение карты глубины
# gray_left = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
# gray_right = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
# stereo = cv2.StereoBM_create(numDisparities=0, blockSize=15)
# disparity = stereo.compute(gray_left, gray_right)
# plt.imshow(disparity, 'gray')
# plt.show()
# img3 = cv2.drawMatches(image_1, key_points_1, image_2, key_points_2, good, image_1, flags=2)
#
# plt.imshow(img3)
# plt.show()
