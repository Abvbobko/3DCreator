import creator_3d.reconstuctor.actions.action as actions
from creator_3d.reconstuctor.constants import algorithm_default_params
import cv2


class SIFT(actions.Extract):

    __default_params = algorithm_default_params.SIFT_DEFAULT_PARAMS

    def __init__(self, **kwargs):
        """
        Args:
            default_params_dict: dict with default parameters value
            **kwargs: list of current algorithm parameters
        """

        # test: check super() with abstract init
        super().__init__(**kwargs)
        params = self.__generate_params_dict(**kwargs)
        self.sift = self.__get_sift_with_params(**params)

    @staticmethod
    def __get_sift_with_params(params_dict):
        return cv2.SIFT_create(**params_dict)

    def detect_and_compute(self, image, mask=None):
        """Find key points and descriptors"""
        return self.sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask)

    def __str__(self):
        return "SIFT"


class SURF(actions.Extract):
    __default_params = algorithm_default_params.SURF_DEFAULT_PARAMS

    def __init__(self, **kwargs):
        """
        Args:
            default_params_dict: dict with default parameters value
            **kwargs: list of current algorithm parameters
        """

        super().__init__(**kwargs)

        params = self.__generate_params_dict(**kwargs)
        self.surf = self.__get_surf_object(**params)

    @staticmethod
    def __get_surf_object(params_dict):
        # todo: check
        return cv2.SURF_create(**params_dict)

    def detect_and_compute(self, image, mask=None):
        """Find key points and descriptors"""
        return self.surf.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask)

    def __str__(self):
        return "SURF"


class ORB(actions.Extract):
    __default_params = algorithm_default_params.SURF_DEFAULT_PARAMS

    def __init__(self, **kwargs):
        """
        Args:
            default_params_dict: dict with default parameters value
            **kwargs: list of current algorithm parameters
        """

        super().__init__(**kwargs)

        params = self.__generate_params_dict(**kwargs)
        self.orb = self.__get_orb_object(**params)

    @staticmethod
    def __get_orb_object(params_dict):
        return cv2.ORB_create(**params_dict)

    def detect_and_compute(self, image, mask=None):
        """Find key points and descriptors"""
        return self.orb.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask)

    def __str__(self):
        return "ORB"


