from creator_3d.reconstuctor.actions.action import Action
from creator_3d.reconstuctor.constants import algorithm_default_params
import cv2
import numpy as np


class SIFT(Action):
    def __init__(self, action_name="SIFT", default_params_dict=None, **kwargs):
        """
        Args:
            action_name: name of the algorithm
            default_params_dict: dict with default parameters value
            **kwargs: list of current algorithm parameters
        """
        if not default_params_dict:
            default_params_dict = algorithm_default_params.SIFT_DEFAULT_PARAMS.copy()
        super(SIFT, self).__init__(action_name, default_params_dict, **kwargs)

        params = self.__generate_sift_params_dict(**kwargs)
        self.sift = self.__get_sift_with_params(**params)

    def __generate_sift_params_dict(self, **kwargs):
        """Get params for algorithm from kwargs"""
        params = self.get_param_names()
        result_param_dict = {}
        for param in params:
            if param in kwargs:
                result_param_dict[param] = kwargs.get(param)
        return result_param_dict

    @staticmethod
    def __get_sift_with_params(params_dict):
        return cv2.SIFT_create(**params_dict)

    def reset_params(self):
        """Set params to default values"""
        new_params = None
        if self.__default_parameters:
            new_params = self.__default_parameters.copy()
        self.__params = new_params

    def run(self, **kwargs):
        image = kwargs['image']
        # todo: создавать словарь для следующего шага
        result_temp = self.__extract_features(image)
        return result_temp


class SURF(Action):
    pass


class ORB(Action):
    pass

