from creator_3d.reconstuctor.kernel.pipeline import Pipeline
from creator_3d.reconstuctor.actions.step import StepAlgorithmParamsWrapper, Step
from creator_3d.reconstuctor.camera_calibration import Camera
from creator_3d.data_access.data_controller import DataController


class Model:
    def __init__(self, model, error):
        self.__model = model
        self.__error = error

    @property
    def error(self):
        return self.__error

    @property
    def model(self):
        return self.__model


class Kernel:
    @staticmethod
    def get_algorithm_object(step, algorithm_name, algorithm_params):
        return step.get_algorithm_object(algorithm_name, **algorithm_params)

    @staticmethod
    def run_pipeline(image_dir: str, image_names: list, camera: Camera,
                     extract_step: Step, extract_algorithm_params: StepAlgorithmParamsWrapper,
                     match_step: Step, match_algorithm_params: StepAlgorithmParamsWrapper,
                     reconstruct_step: Step, reconstruct_algorithm_params: StepAlgorithmParamsWrapper,
                     bundle_adjust_step: Step, bundle_adjust_algorithm_params: StepAlgorithmParamsWrapper):

        # create step objects
        extractor = Kernel.get_algorithm_object(extract_step,
                                                extract_algorithm_params.algorithm_name,
                                                extract_algorithm_params.params)
        matcher = Kernel.get_algorithm_object(match_step,
                                              match_algorithm_params.algorithm_name,
                                              match_algorithm_params.params)
        reconstructor = Kernel.get_algorithm_object(reconstruct_step,
                                                    reconstruct_algorithm_params.algorithm_name,
                                                    reconstruct_algorithm_params.params)
        bundle_adjuster = Kernel.get_algorithm_object(bundle_adjust_step,
                                                      bundle_adjust_algorithm_params.algorithm_name,
                                                      bundle_adjust_algorithm_params.params)

        pipeline = Pipeline(feature_extractor=extractor,
                            feature_matcher=matcher,
                            reconstructor=reconstructor,
                            bundle_adjuster=bundle_adjuster,
                            camera=camera)

        image_paths = DataController.get_full_image_paths(image_dir, image_names)
        process_result = pipeline.run(image_paths)
        if isinstance(process_result, str):
            return Model(None, error=process_result)
        return Model(process_result, None)

