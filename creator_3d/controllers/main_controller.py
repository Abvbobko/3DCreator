from creator_3d.reconstuctor.connector import ReconstructorConnector
from creator_3d.data_access.data_controller import DataController
from creator_3d.reconstuctor.camera_calibration import Camera


class MainController:
    def __init__(self):
        self.reconstructor = ReconstructorConnector()
        self.data_accessor = DataController()

    def get_params_from_exif_image(self, image_path) -> Camera:
        return self.reconstructor.get_camera_params_from_exif(image_path)

    def load_images_by_paths(self, image_paths):
        images = []
        for image_path in image_paths:
            images.append(self.load_image_by_path(image_path))
        return images

    def load_image_by_path(self, image_path):
        return self.data_accessor.read_cv2_image(image_path)

    def get_step_algorithms(self, step_name):
        return self.reconstructor.get_algorithms_for_step(step_name)

    def get_extract_step_name(self):
        return self.reconstructor.get_feature_extract_step_name()

    def get_match_step_name(self):
        return self.reconstructor.get_feature_match_step_name()

    def get_reconstruct_step_name(self):
        return self.reconstructor.get_reconstruct_step_name()

    def get_bundle_adjust_step_name(self):
        return self.reconstructor.get_bundle_adjust_step_name()
