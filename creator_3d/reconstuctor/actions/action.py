from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class Action(ABC):

    __default_params = {}

    def __init__(self, **kwargs):
        self.__params = kwargs

    @property
    def get_default_params(self):
        """Get default action parameters."""
        return self.__default_params.copy()

    def get_default_param_by_name(self, param_name):
        """Get default param value by name"""
        return self.__default_params.get(param_name)

    def get_param_names(self):
        """Get list of all parameter names."""
        return list(self.__default_params.keys())

    def get_params(self):
        """Get all parameters dict."""
        if self.__params:
            return self.__params.copy()
        return None

    def get_param_value_by_name(self, param_name, default=None):
        """Get param value by name.
        If there is no parameter with name -> return None."""
        return self.__params.get(param_name, default)

    def set_param_by_name(self, param_name, value):
        """Set new value to parameter."""
        self.__params[param_name] = value

    def set_new_params(self, param_dict):
        """Set new values to params.

        Args:
            param_dict (dict): dict where key is param name
            and value is new param value.
        """

        self.__params.update(param_dict)

    def reset_params(self):
        """Set params to default values"""
        self.__params = self.__default_params.copy()

    def __generate_params_dict(self, **kwargs):
        """Get params for algorithm from kwargs"""
        params = self.get_param_names()
        result_param_dict = {}
        for param in params:
            if param in kwargs:
                result_param_dict[param] = kwargs.get(param)
        return result_param_dict


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
    def bundle_adjustment(self, **params):
        """Combine points to points cloud."""
        pass
