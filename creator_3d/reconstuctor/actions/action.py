from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


# todo: add steps? and then наследоваться от шага

class Action(ABC):
    def __init__(self, action_name='', default_params_dict=None, **kwargs):
        self._action_name = action_name
        self.__params = kwargs
        self.__default_parameters = default_params_dict

    @property
    def action_name(self):
        """Human readable name of action (algorithm) like SIFT"""
        if not self._action_name:
            logger.error("Action does not have action_name.")
        return self._action_name

    @property
    def get_default_params(self):
        """Get default action parameters."""
        if self.__default_parameters:
            return self.__default_parameters.copy()
        return None

    def get_default_param_by_name(self, param_name):
        """Get default param value by name"""
        return self.__default_parameters.get(param_name)

    def get_param_names(self):
        """Get list of all parameter names."""
        return list(self.__default_parameters.keys())

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

    @abstractmethod
    def reset_params(self):
        """Reset all parameters to default."""
        pass

    # @abstractmethod
    # def run(self, **kwargs):
    #     """Start action method and
    #     return result dict that contains parameters for the next step.
    #     """
    #     pass


class Extract(Action, ABC):
    @abstractmethod
    def detect_and_compute(self, image, mask):
        """Find key points and descriptors"""
        pass


