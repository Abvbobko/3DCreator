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

    def __init__(self):
        self.extractor = self.get_step_by_name(steps_const.EXTRACT_DEFAULT_PARAMS.name)
        self.matcher = self.get_step_by_name(steps_const.MATCH_DEFAULT_PARAMS.name)
        self.reconstructor = self.get_step_by_name(steps_const.RECONSTRUCT_DEFAULT_PARAMS.name)
        self.bundle_adjuster = self.get_step_by_name(steps_const.BUNDLE_ADJUST_DEFAULT_PARAMS.name)
        # self.camera = None
        self.kernel = Kernel()

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
