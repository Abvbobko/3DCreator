"""This code just for run other code in test mode
"""

from pipelines import feature_extraction
from pipelines import feature_matching
from pipelines import geometric_verification
import cv2
import matplotlib.pyplot as plt
import numpy as np


sift = feature_extraction.SIFT()
matcher = feature_matching.BFMatcher()
geometric_verifier = geometric_verification.PointForFMatrix()


data_path = "C:\\Users\\hp\\Desktop\\3DCreator\\3d_creator\\data\\sculpture\\"
image_1_path = data_path + "2.jpg"
image_2_path = data_path + "3.jpg"

image_1 = cv2.imread(image_1_path)
image_2 = cv2.imread(image_2_path)

key_points_1, descriptors_1 = sift.extract_features(image_1)
key_points_2, descriptors_2 = sift.extract_features(image_2)

good = matcher.match_features(key_points_1, descriptors_1, key_points_2, descriptors_2)

geometric_verifier.verify(good, image_1, image_2, key_points_1, key_points_2)

# img3 = cv2.drawMatches(image_1, key_points_1, image_2, key_points_2, good, image_1, flags=2)

# plt.imshow(img3)
# plt.show()
