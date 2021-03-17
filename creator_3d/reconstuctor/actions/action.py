from abc import ABC


class Action(ABC):
    def run(self, **kwargs):
        """Start action method"""
        pass

    def get_result_dict(self):
        """Return result dict that contains parameters for the next step."""
        pass
