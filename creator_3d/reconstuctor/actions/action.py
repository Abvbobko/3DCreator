from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class Action(ABC):
    _default_params = {}
    _name = None

    @abstractmethod
    def __init__(self, **params):
        pass

    @staticmethod
    def convert_type(value, new_type):
        return new_type(value)

    @classmethod
    def validate_params(cls, **params):
        for param_name in cls._default_params:
            if param_name not in params:
                return f"Parameter {param_name} is not set ({cls.name()})."

            default_type = type(cls._default_params[param_name])
            if not isinstance(params[param_name], default_type):
                try:
                    default_type(params[param_name])
                except ValueError:
                    return f"Type of {param_name} should be {type(cls._default_params[param_name])} ({cls.name()})."
            if default_type is bool and (isinstance(params[param_name], str)
                                         and params[param_name].lower() not in ["false", "true", "t", "f", "0", "1"]):
                return f"Type of {param_name} should be {type(cls._default_params[param_name])} ({cls.name()})."
        return None

    @classmethod
    def convert_params_type_from_str(cls, **params):
        new_params = {}
        for param_name in cls._default_params:
            default_type = type(cls._default_params[param_name])
            if not isinstance(params[param_name], default_type):
                converted_value = default_type(params[param_name])
                new_params[param_name] = converted_value
            else:
                new_params[param_name] = params[param_name]

            if default_type is bool and (isinstance(params[param_name], str)
                                         and params[param_name].lower() in ["false", "true", "t", "f", "0", "1"]):
                converted_value = default_type(params[param_name])
                new_params[param_name] = converted_value
        return new_params

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

    def _generate_params_dict(self, **kwargs):
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

    @abstractmethod
    def find_transform(self, K, p1, p2):
        pass


class BundleAdjustment(Action, ABC):
    @abstractmethod
    def adjust_bundle(self, **params):
        """Combine points to points cloud."""
        pass
