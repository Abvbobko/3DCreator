import creator_3d.reconstuctor.actions.action as actions
from creator_3d.reconstuctor.constants import algorithm_default_params
import cv2


# todo: add all docstrings?

class SIFT(actions.Extract):
    def __init__(self, default_params_dict=None, **kwargs):
        """
        Args:
            default_params_dict: dict with default parameters value
            **kwargs: list of current algorithm parameters
        """
        if not default_params_dict:
            default_params_dict = algorithm_default_params.SIFT_DEFAULT_PARAMS.copy()
        super(SIFT, self).__init__(default_params_dict, **kwargs)

        params = self.__generate_params_dict(**kwargs)
        self.sift = self.__get_sift_with_params(**params)

    @staticmethod
    def __get_sift_with_params(params_dict):
        return cv2.SIFT_create(**params_dict)

    def reset_params(self):
        """Set params to default values"""
        new_params = None
        if self.__default_parameters:
            new_params = self.__default_parameters.copy()
        self.__params = new_params

    def detect_and_compute(self, image, mask=None):
        """Find key points and descriptors"""
        return self.sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask)

    def __str__(self):
        return "SIFT"


class SURF(actions.Extract):
    pass


class ORB(actions.Extract):
    pass

