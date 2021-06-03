from creator_3d.reconstuctor.actions.steps import feature_extraction, feature_matching, reconstruction, \
    bundle_adjustment

from creator_3d.reconstuctor.actions.step import Step
from creator_3d.reconstuctor.constants import steps_default_params


class ActionController:

    def __init__(self):

        # init steps
        self.feature_extraction = Step(
            feature_extraction,
            step_name=steps_default_params.EXTRACT_DEFAULT_PARAMS.name,
            default_algorithm_name=steps_default_params.EXTRACT_DEFAULT_PARAMS.default_algorithm
        )

        self.feature_matching = Step(
            feature_matching,
            step_name=steps_default_params.MATCH_DEFAULT_PARAMS.name,
            default_algorithm_name=steps_default_params.MATCH_DEFAULT_PARAMS.default_algorithm
        )

        self.reconstruction = Step(
            reconstruction,
            step_name=steps_default_params.RECONSTRUCT_DEFAULT_PARAMS.name,
            default_algorithm_name=steps_default_params.RECONSTRUCT_DEFAULT_PARAMS.default_algorithm
        )

        self.bundle_adjustment = Step(
            bundle_adjustment,
            step_name=steps_default_params.BUNDLE_ADJUST_DEFAULT_PARAMS.name,
            default_algorithm_name=steps_default_params.BUNDLE_ADJUST_DEFAULT_PARAMS.default_algorithm
        )

    def get_feature_extraction_algorithms(self):
        """Get all feature extraction algorithms"""
        return self.get_step_algorithms(self.feature_extraction)

    def get_feature_matching_algorithms(self):
        """Get all feature matching algorithms"""
        return self.get_step_algorithms(self.feature_matching)

    def get_reconstruction_algorithms(self):
        """Get all reconstruction algorithms"""
        return self.get_step_algorithms(self.reconstruction)

    def get_bundle_adjustment_algorithms(self):
        """Get all bundle adjustment algorithms"""
        return self.get_step_algorithms(self.bundle_adjustment)

    @staticmethod
    def get_step_algorithms(step: Step):
        """Get all algorithms of the step"""
        return step.get_step_algorithms()

    def get_step_by_name(self, step_name):
        """Get step object by name

        Args:
            step_name (str): name of the step

        Returns:
            (Step): step object with step_name
        """

        steps = [self.feature_extraction, self.feature_matching, self.reconstruction, self.bundle_adjustment]
        for step in steps:
            if step_name == str(step):
                return step
        return None
