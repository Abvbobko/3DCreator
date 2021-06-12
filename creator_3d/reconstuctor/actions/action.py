from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class Action(ABC):

    _default_params = {}
    _name = None

    @abstractmethod
    def __init__(self, **params):
        pass

    @classmethod
    def get_default_params(cls):
        """Get default action parameters."""
        return cls._default_params.copy()

    def get_default_param_by_name(self, param_name):
        """Get default param value by name"""
        return self._default_params.get(param_name)

    def get_param_names(self):
        """Get list of all parameter names."""
        return list(self._default_params.keys())

    def __generate_params_dict(self, **kwargs):
        """Get params for algorithm from kwargs"""
        params = self.get_param_names()
        result_param_dict = {}
        for param in params:
            if param in kwargs:
                result_param_dict[param] = kwargs.get(param)
        return result_param_dict

    @classmethod
    def name(cls):
        return cls._name


class Extract(Action, ABC):
    @abstractmethod
    def detect_and_compute(self, image, mask=None):
        """Find key points and descriptors"""
        pass


class Match(Action, ABC):
    @abstractmethod
    def match_features(self, descriptors_img1, descriptors_img2):
        """Match features of two images."""
        pass


class Reconstruct(Action, ABC):
    @abstractmethod
    def reconstruct(self, **params):
        """Reconstruct 3D points."""
        pass


class BundleAdjustment(Action, ABC):
    @abstractmethod
    def adjust_bundle(self, **params):
        """Combine points to points cloud."""
        pass
