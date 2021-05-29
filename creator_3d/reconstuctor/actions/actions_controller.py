from creator_3d.reconstuctor.actions.steps import feature_extraction, feature_matching, reconstruction, \
    bundle_adjustment

from creator_3d.reconstuctor.actions.step import Step


class ActionController:

    def __init__(self):
        self.feature_extraction = Step(feature_extraction, step_name="feature extraction")
        self.feature_matching = Step(feature_matching, step_name="feature matching")
        self.reconstruction = Step(reconstruction, step_name="reconstruction")
        self.bundle_adjustment = Step(bundle_adjustment, step_name="bundle adjustment")

    def get_feature_extraction_algorithms(self):
        return self.get_step_algorithms(self.feature_extraction)

    def get_feature_matching_algorithms(self):
        return self.get_step_algorithms(self.feature_matching)

    def get_reconstruction_algorithms(self):
        return self.get_step_algorithms(self.reconstruction)

    def get_bundle_adjustment_algorithms(self):
        return self.get_step_algorithms(self.bundle_adjustment)

    @staticmethod
    def get_step_algorithms(step: Step):
        return step.get_step_algorithms()

    def get_feature_extraction_default_params(self):
        return self.get_algorithm_default_params(self.feature_extraction)

    def get_feature_matching_default_params(self):
        return self.get_algorithm_default_params(self.feature_matching)

    def get_reconstruction_default_params(self):
        return self.get_algorithm_default_params(self.reconstruction)

    def get_bundle_adjustment_default_params(self):
        return self.get_algorithm_default_params(self.bundle_adjustment)

    @staticmethod
    def get_algorithm_default_params(algorithm: Step):
        return algorithm.get_current_algorithm_default_params()

