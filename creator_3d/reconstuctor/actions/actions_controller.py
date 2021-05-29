import inspect
import sys

from creator_3d.reconstuctor.actions.steps import feature_extraction, feature_matching, reconstruction, \
    bundle_adjustment


class StepController:
    def __init__(self, step_module, step_name):
        self.module = step_module
        self.name = step_name

    def get_step_algorithms(self):
        algorithms = inspect.getmembers(sys.modules[self.module.__name__], inspect.isclass)
        return [class_name for class_name, _ in algorithms]

    def __str__(self):
        return self.name


class ActionController:

    def __init__(self):
        self.feature_extraction = StepController(feature_extraction, step_name="feature extraction")
        self.feature_matching = StepController(feature_matching, step_name="feature matching")
        self.reconstruction = StepController(reconstruction, step_name="reconstruction")
        self.bundle_adjustment = StepController(bundle_adjustment, step_name="bundle adjustment")

    def get_feature_extraction_algorithms(self):
        return self.get_step_algorithms(self.feature_extraction)

    def get_feature_matching_algorithms(self):
        return self.get_step_algorithms(self.feature_matching)

    def get_reconstruction_algorithms(self):
        return self.get_step_algorithms(self.reconstruction)

    def get_bundle_adjustment_algorithms(self):
        return self.get_step_algorithms(self.bundle_adjustment)

    @staticmethod
    def get_step_algorithms(step: StepController):
        return step.get_step_algorithms()
