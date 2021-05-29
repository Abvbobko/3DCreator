import inspect
import sys

from creator_3d.reconstuctor.actions.action import Action


class Step:
    def __init__(self, step_module, step_name):
        self.module = step_module
        self.name = step_name
        self.current_algorithm = None

    def get_step_algorithms(self):
        algorithms = inspect.getmembers(sys.modules[self.module.__name__], inspect.isclass)
        return [class_name for class_name, _ in algorithms]

    @staticmethod
    def get_algorithm_default_params(algorithm: Action):
        return algorithm.get_default_params

    def set_current_algorithm(self, algorithm_class, **params):
        self.current_algorithm = algorithm_class(**params)

    @staticmethod
    def get_algorithm_params(algorithm: Action):
        if algorithm:
            return algorithm.get_params()
        return None

    def get_current_algorithm_params(self):
        return self.get_algorithm_params(self.current_algorithm)

    def get_current_algorithm_default_params(self):
        if self.current_algorithm:
            return self.get_algorithm_params(self.current_algorithm)
        return None

    def get_current_algorithm(self):
        return self.current_algorithm

    def __str__(self):
        return self.name
