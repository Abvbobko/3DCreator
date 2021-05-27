from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class Action(ABC):
    def __init__(self, action_name='', **kwargs):
        self._action_name = action_name
        self.__params = kwargs

    @property
    def action_name(self):
        """Human readable name of action (algorithm) like SIFT"""
        if not self._action_name:
            logger.error("Action does not have action_name.")
        return self._action_name

    def get_param_names(self):
        """Get list of all parameter names."""
        return list(self.__params.keys())

    def get_params(self):
        """Get all parameters dict."""
        return self.__params.copy()

    def get_param_value_by_name(self, param_name, default=None):
        """Get param value by name.
        If there is no parameter with name -> return None."""
        return self.__params.get(param_name, default)

    def set_param_by_name(self, param_name, value):
        """Set new value to parameter."""
        self.__params[param_name] = value

    @abstractmethod
    def reset_params(self):
        """Reset all parameters to default."""
        pass

    @abstractmethod
    def run(self, **kwargs):
        """Start action method and
        return result dict that contains parameters for the next step.
        """
        pass

