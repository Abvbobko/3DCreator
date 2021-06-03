from creator_3d.reconstuctor.actions.action import Match

from abc import ABC, abstractmethod
import cv2
import numpy as np
from creator_3d.reconstuctor.constants import algorithm_default_params
from creator_3d.reconstuctor.constants import pipeline_const

# todo: есть knnMatch, а есть просто match узнать, в чем разница


class BFMatcher(Match):

    __default_params = algorithm_default_params.BF_DEFAULT_PARAMS

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        params = self.__generate_params_dict(**kwargs)
        self.bf = self.__get_bf_object(**params)

    def match_features(self, query, train):
        knn_matches = self.bf.knnMatch(query, train, k=2)  # todo: вынести в параметры
        matches = []
        for m, n in knn_matches:
            if m.distance < pipeline_const.MRT * n.distance:
                matches.append(m)

        return np.array(matches)

    @staticmethod
    def __get_bf_object(params_dict):
        return cv2.BFMatcher(**params_dict)

    def __str__(self):
        return "BF"


class FLANNMatcher(Match):

    __default_params = algorithm_default_params.FLANN_DEFAULT_PARAMS

    def __init__(self, **kwargs):
        super(FLANNMatcher, self).__init__(**kwargs)

        params = self.__generate_params_dict(**kwargs)
        self.bf = self.__get_flann_object(**params)

    def match_features(self, query, train):
        knn_matches = self.bf.knnMatch(query, train, k=2)
        matches = []
        for m, n in knn_matches:
            if m.distance < pipeline_const.MRT * n.distance:
                matches.append(m)

        return np.array(matches)

    @staticmethod
    def __get_flann_object(params_dict):
        return cv2.FlannBasedMatcher(**params_dict)

    def __str__(self):
        return "FLANN"


# class FLANNMatcher(Match):
#     def reset_params(self):
#         pass
#
#     def __init__(self, action_name="FLANN", default_params_dict=None, **kwargs):
#         if not default_params_dict:
#             default_params_dict = algorithm_default_params.FLANN_DEFAULT_PARAMS.copy()
#         super(FLANNMatcher, self).__init__(action_name, default_params_dict, **kwargs)
#
#         params = self.__generate_params_dict(**kwargs)
#         self.flann = self.__get_flann_with_params(**params)
#
#     def match_features(self, query, train):
#         bf = cv2.BFMatcher(cv2.NORM_L2) # todo: убрать метчер
#         knn_matches = bf.knnMatch(query, train, k=2) # todo: вынести в параметры
#         matches = []
#         for m, n in knn_matches:
#             if m.distance < pipeline_const.MRT * n.distance:
#                 matches.append(m)
#
#         return np.array(matches)
#
#     @staticmethod
#     def __get_flann_with_params(params_dict):
#         index_params = dict(algorithm=params_dict.get("index_kdtree"),
#                             trees=params_dict.get("trees"))
#
#         search_params = dict(params_dict.get("check"))
#         return cv2.FlannBasedMatcher(index_params, search_params)

# class BFMatcher(Action):
#     def __init__(self, action_name="BF"):
#         # todo: change action_name
#         self.__action_name = action_name
#         self.bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
#
#     # def match_features(self, key_points_img1, descriptors_img1, key_points_img2, descriptors_img2):
#     #     # Match descriptors.
#     #     matches = self.bf.knnMatch(descriptors_img1, descriptors_img2, k=2)
#     #
#     #     # Apply ratio test
#     #     good = []
#     #     for m, n in matches:
#     #         if m.distance < 0.75 * n.distance:
#     #             good.append(m)
#     #
#     #     return good
#
#     def match_features(self, query, train):
#
#         bf = cv2.BFMatcher(cv2.NORM_L2)
#         knn_matches = bf.knnMatch(query, train, k=2)
#         matches = []
#         for m, n in knn_matches:
#             # todo: вынести в параметры
#             if m.distance < const.MRT * n.distance:
#                 matches.append(m)
#
#         return np.array(matches)
#
#     @property
#     def action_name(self):
#         return self.__action_name
#
#     def run(self, **kwargs):
#         pass
