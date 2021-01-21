from abc import ABC, abstractmethod
import cv2
import numpy as np


class FeatureMatcher(ABC):
    @abstractmethod
    def match_features(self, key_points_img1, descriptors_img1, key_points_img2, descriptors_img2):
        pass


class FLANNMatcher(FeatureMatcher):
    def __init__(self):
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1

        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

        search_params = dict(checks=50) # or pass empty dictionary
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def match_features(self, key_points_img1, descriptors_img1, key_points_img2, descriptors_img2):

        matches = self.flann.knnMatch(descriptors_img1, descriptors_img2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        return good_matches


class BFMatcher(FeatureMatcher):
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match_features(self, key_points_img1, descriptors_img1, key_points_img2, descriptors_img2):
        # Match descriptors.
        matches = self.bf.match(descriptors_img1, descriptors_img2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
