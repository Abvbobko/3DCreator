from creator_3d.reconstuctor.actions.actions_controller import ActionController
from creator_3d.reconstuctor.actions.action import Action
from creator_3d.data_access.data_controller import DataController
from creator_3d.reconstuctor.camera_calibration import Calibrator, Camera
from creator_3d.reconstuctor.kernel.kernel import Kernel
from creator_3d.reconstuctor.actions.step import Step
from creator_3d.reconstuctor.constants import steps_default_params as steps_const


class ReconstructorConnector:

    action_controller = ActionController()
    data_controller = DataController()
    camera_calibrator = Calibrator()
    kernel = Kernel()

    def __init__(self):
        self.extractor = self.get_step_by_name(steps_const.EXTRACT_DEFAULT_PARAMS.name)
        self.matcher = self.get_step_by_name(steps_const.MATCH_DEFAULT_PARAMS.name)
        self.reconstructor = self.get_step_by_name(steps_const.RECONSTRUCT_DEFAULT_PARAMS.name)
        self.bundle_adjuster = self.get_step_by_name(steps_const.BUNDLE_ADJUST_DEFAULT_PARAMS.name)

    def get_step_by_name(self, step_name):
        return self.action_controller.get_step_by_name(step_name)

    def process_images(self, image_paths):
        # todo: implement
        pass

    @staticmethod
    def get_algorithm_default_parameters(algorithm: Action):
        return algorithm.get_default_params()

    @staticmethod
    def get_step_default_algorithm(step: Step):
        return step.get_default_step_algorithm_name()

    def get_algorithms_for_step(self, step_name):
        steps = self.action_controller.get_steps()
        for step in steps:
            if str(step) == step_name:
                step_object = self.action_controller.get_step_by_name(step_name)
                return step_object.get_step_algorithms()
        return None

    def get_algorithm_names_for_step(self, step_name):
        algorithms = self.get_algorithms_for_step(step_name)
        return [str(algorithm) for algorithm in algorithms]

    def get_camera_params_from_exif(self, image_path) -> Camera:
        image = self.data_controller.read_pil_image(image_path)
        camera = self.camera_calibrator.get_camera_params_from_exif(image)
        return camera

    def get_camera_object(self, focal_length, sensor_size, image_size):
        return self.camera_calibrator.get_camera_object(focal_length=focal_length,
                                                        sensor_size=sensor_size,
                                                        image_size=image_size)

    @staticmethod
    def get_step_name(step: Step):
        return str(step)

    def get_feature_extract_step_name(self):
        return self.get_step_name(self.extractor)

    def get_feature_match_step_name(self):
        return self.get_step_name(self.matcher)

    def get_reconstruct_step_name(self):
        return self.get_step_name(self.reconstructor)

    def get_bundle_adjust_step_name(self):
        return self.get_step_name(self.bundle_adjuster)

    def get_feature_extract_default_algorithm(self):
        return self.get_step_default_algorithm(self.extractor)

    def get_feature_match_default_algorithm(self):
        return self.get_step_default_algorithm(self.matcher)

    def get_reconstruct_default_algorithm(self):
        return self.get_step_default_algorithm(self.reconstructor)

    def get_bundle_adjust_default_algorithm(self):
        return self.get_step_default_algorithm(self.bundle_adjuster)

    @staticmethod
    def get_algorithm_default_params(algorithm):
        return algorithm.get_default_params()

    def get_step_algorithm_default_params(self, step_name, algorithm_name=''):
        step = self.get_step_by_name(step_name)
        if not algorithm_name:
            algorithm_name = step.get_default_step_algorithm_name()

        algorithm_class = step.get_algorithm_class_by_name(algorithm_name)
        if algorithm_class:
            return algorithm_class.get_default_params()
        return {}

    def validate_step_algorithm_params(self, step_name, algorithm_name, **params):
        return self.action_controller.validate_step_algorithm_params(step_name, algorithm_name, **params)

    def wrap_step_algorithm_params(self, step_name, algorithm_name, params):
        return self.action_controller.wrap_step_params(step_name, algorithm_name, params)

    def process(self, camera, step_algorithms, image_dir, image_names):
        return self.kernel.run_pipeline(image_dir=image_dir,
                                        image_names=image_names,
                                        camera=camera,
                                        extract_step=self.extractor,
                                        extract_algorithm_params=step_algorithms[str(self.extractor)],
                                        match_step=self.matcher,
                                        match_algorithm_params=step_algorithms[str(self.matcher)],
                                        reconstruct_step=self.reconstructor,
                                        reconstruct_algorithm_params=step_algorithms[str(self.reconstructor)],
                                        bundle_adjust_step=self.bundle_adjuster,
                                        bundle_adjust_algorithm_params=step_algorithms[str(self.bundle_adjuster)])
