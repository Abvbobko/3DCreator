from creator_3d.reconstuctor.actions.action import Action

from abc import ABC, abstractmethod
import cv2
import numpy as np


# есть knnMatch, а есть просто match
# узнать, в чем разница
#

# class FeatureMatcher(ABC):
#     @abstractmethod
#     def match_features(self, key_points_img1, descriptors_img1, key_points_img2, descriptors_img2):
#         pass

class FLANNMatcher(Action):
    def __init__(self, action_name="FLANN"):

        self.__action_name = action_name
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

    @property
    def action_name(self):
        return self.__action_name

    def run(self, **kwargs):
        pass


class BFMatcher(Action):
    def __init__(self, action_name="BF"):
        # todo: change action_name
        self.__action_name = action_name
        self.bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

    # def match_features(self, key_points_img1, descriptors_img1, key_points_img2, descriptors_img2):
    #     # Match descriptors.
    #     matches = self.bf.knnMatch(descriptors_img1, descriptors_img2, k=2)
    #
    #     # Apply ratio test
    #     good = []
    #     for m, n in matches:
    #         if m.distance < 0.75 * n.distance:
    #             good.append(m)
    #
    #     return good

    def match_features(self, query, train):

        bf = cv2.BFMatcher(cv2.NORM_L2)
        knn_matches = bf.knnMatch(query, train, k=2)
        matches = []
        for m, n in knn_matches:
            # todo: вынести в параметры
            if m.distance < const.MRT * n.distance:
                matches.append(m)

        return np.array(matches)

    @property
    def action_name(self):
        return self.__action_name

    def run(self, **kwargs):
        pass
