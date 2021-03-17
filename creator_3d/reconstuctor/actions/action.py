from abc import ABC, abstractmethod


class Action(ABC):
    @abstractmethod
    def run(self, **kwargs):
        """Start action method and
        return result dict that contains parameters for the next step.
        """

        pass

    # @abstractmethod
    # def get_result_dict(self):
    #     """Return result dict that contains parameters for the next step."""
    #     pass

    @property
    @abstractmethod
    def action_name(self):
        """Human readable name of action (algorithm) like SIFT"""
        pass
