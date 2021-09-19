import creator_3d.reconstuctor.actions.action as actions
from creator_3d.reconstuctor.constants import algorithm_default_params as default_params
import cv2


class SIFT(actions.Extract):

    # todo: think about default params and generalize logic?
    _default_params = default_params.SIFT_DEFAULT_PARAMS.params
    _name = default_params.SIFT_DEFAULT_PARAMS.name

    def __init__(self, **kwargs):
        """
        Args:
            default_params_dict: dict with default parameters value
            **kwargs: list of current algorithm parameters
        """

        # test: check super() with abstract init
        super().__init__(**kwargs)
        params = self._generate_params_dict(**kwargs)
        self.sift = self.__get_sift_with_params(**params)

    @staticmethod
    def __get_sift_with_params(**params_dict):
        return cv2.SIFT_create(**params_dict)

    def detect_and_compute(self, image, mask=None):
        """Find key points and descriptors"""
        return self.sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask)


class SURF(actions.Extract):
    _default_params = default_params.SURF_DEFAULT_PARAMS.params
    _name = default_params.SURF_DEFAULT_PARAMS.name

    def __init__(self, **kwargs):
        """
        Args:
            default_params_dict: dict with default parameters value
            **kwargs: list of current algorithm parameters
        """

        super().__init__(**kwargs)

        params = self._generate_params_dict(**kwargs)
        self.surf = self.__get_surf_object(**params)

    @staticmethod
    def __get_surf_object(**params_dict):
        # todo: check
        return cv2.SURF_create(**params_dict)

    def detect_and_compute(self, image, mask=None):
        """Find key points and descriptors"""
        return self.surf.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask)


class ORB(actions.Extract):
    _default_params = default_params.ORB_DEFAULT_PARAMS.params
    _name = default_params.ORB_DEFAULT_PARAMS.name

    def __init__(self, **kwargs):
        """
        Args:
            default_params_dict: dict with default parameters value
            **kwargs: list of current algorithm parameters
        """

        super().__init__(**kwargs)

        params = self._generate_params_dict(**kwargs)
        self.orb = self.__get_orb_object(**params)

    @staticmethod
    def __get_orb_object(**params_dict):
        return cv2.ORB_create(**params_dict)

    def detect_and_compute(self, image, mask=None):
        """Find key points and descriptors"""
        return self.orb.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask)


