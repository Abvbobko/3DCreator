"""This code just for run other code in test mode
"""

from pipelines import feature_extraction
from pipelines import feature_matching
import cv2


sift = feature_extraction.SIFT()
matcher = feature_matching.FLANNMatcher()

image_1_path = "C:\\Users\\hp\\Desktop\\3DCreator\\3d_creator\\data\\sculpture\\2.jpg"
image_2_path = "C:\\Users\\hp\\Desktop\\3DCreator\\3d_creator\\data\\sculpture\\3.jpg"

image_1 = cv2.imread(image_1_path)
image_2 = cv2.imread(image_2_path)

key_points_1, descriptors_1 = sift.extract_features(image_1)
key_points_2, descriptors_2 = sift.extract_features(image_2)

matcher.match_features(key_points_1, descriptors_1, key_points_2, descriptors_2)