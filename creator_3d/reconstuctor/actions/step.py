import inspect
import sys

from creator_3d.reconstuctor.actions.action import Action


class StepAlgorithmParamsWrapper:
    def __init__(self, step_name, algorithm_name, params):
        self.__step_name = step_name
        self.__algorithm_name = algorithm_name
        self.__params = params

    @property
    def step_name(self):
        return self.__step_name

    @property
    def algorithm_name(self):
        return self.__algorithm_name

    @property
    def params(self):
        return self.__params


class Step:
    def __init__(self, step_module, step_name, default_algorithm_name):
        self.module = step_module
        self.name = step_name
        self.step_algorithms = self.get_step_algorithms()

        if self.is_algorithm_belongs_to_step(default_algorithm_name):
            self.default_algorithm = default_algorithm_name
        elif self.step_algorithms:
            self.default_algorithm = self.get_step_algorithms()[0]
        else:
            self.default_algorithm = None

    def is_algorithm_belongs_to_step(self, algorithm_name: str):
        """Check if algorithm belongs to step algorithms or not"""
        for algorithm in self.step_algorithms:
            if algorithm_name == algorithm:
                return True
        return False

    def get_algorithm_class_by_name(self, algorithm_name: str):
        for algorithm in self.get_step_algorithm_classes():
            if algorithm_name == algorithm.name():
                return algorithm
        return None

    def get_algorithm_object(self, algorithm_name: str, **params):
        # todo: think maybe we should delete ** everywhere
        algorithm_class = self.get_algorithm_class_by_name(algorithm_name)
        if algorithm_class:
            return algorithm_class(**params)
        return None

    def get_step_algorithms(self):
        """Get all algorithms of step"""
        algorithms = inspect.getmembers(sys.modules[self.module.__name__], inspect.isclass)
        return [class_name for class_name, _ in algorithms]

    def get_step_algorithm_classes(self):
        """Get all algorithms of step"""
        algorithms = inspect.getmembers(sys.modules[self.module.__name__], inspect.isclass)
        return [cls for _, cls in algorithms]

    def get_default_step_algorithm_name(self):
        """Get default algorithm name of the step"""
        return self.default_algorithm

    @staticmethod
    def get_algorithm_default_params(algorithm: Action):
        """Get default params of the algorithm

        Args:
            algorithm (Action): algorithm class
        Returns:
            (dict): dict of algorithm params
        """

        return algorithm.get_default_params()

    def validate_algorithm_params(self, algorithm_name, **params):
        algorithm_class = self.get_algorithm_class_by_name(algorithm_name)
        return algorithm_class.validate_params(**params)

    def __str__(self):
        return self.name
