from creator_3d.reconstuctor.actions.action import Match

from abc import ABC, abstractmethod
import cv2
import numpy as np
from creator_3d.reconstuctor.constants import algorithm_default_params as default_params
from creator_3d.reconstuctor.constants import pipeline_const

# todo: есть knnMatch, а есть просто match узнать, в чем разница


class BFMatcher(Match):

    __default_params = default_params.BF_DEFAULT_PARAMS.params

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
        return default_params.BF_DEFAULT_PARAMS.name


class FLANNMatcher(Match):

    __default_params = default_params.FLANN_DEFAULT_PARAMS.params

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        params = self.__generate_params_dict(**kwargs)
        self.flann = self.__get_flann_object(**params)

    def match_features(self, query, train):
        knn_matches = self.flann.knnMatch(query, train, k=2)
        matches = []
        for m, n in knn_matches:
            if m.distance < pipeline_const.MRT * n.distance:
                matches.append(m)

        return np.array(matches)

    @staticmethod
    def __get_flann_object(params_dict):
        return cv2.FlannBasedMatcher(**params_dict)

    def __str__(self):
        return default_params.FLANN_DEFAULT_PARAMS.name
