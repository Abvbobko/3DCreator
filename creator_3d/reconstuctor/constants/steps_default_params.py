from creator_3d.reconstuctor.constants import algorithm_default_params as algorithms


class StepDefaultParams:
    def __init__(self, step_name: str, default_algorithm_name: str):
        self.__name = step_name
        self.__default_algorithm = default_algorithm_name

    @property
    def name(self):
        return self.__name

    @property
    def default_algorithm(self):
        return self.__default_algorithm


EXTRACT_DEFAULT_PARAMS = StepDefaultParams("feature extraction", algorithms.SIFT_DEFAULT_PARAMS.name)
MATCH_DEFAULT_PARAMS = StepDefaultParams("feature matching", algorithms.BF_DEFAULT_PARAMS.name)
RECONSTRUCT_DEFAULT_PARAMS = StepDefaultParams("reconstruction", algorithms.RECONSTRUCT_DEFAULT_PARAMS.name)
BUNDLE_ADJUST_DEFAULT_PARAMS = StepDefaultParams("bundle adjustment", algorithms.BUNDLE_ADJUSTMENT_DEFAULT_PARAMS.name)
