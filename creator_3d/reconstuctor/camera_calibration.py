import logging
import numpy as np
import PIL.ExifTags
import PIL.Image
# todo: потом убрать отсюда
from creator_3d.data_access.data_controller import DataController

logger = logging.getLogger(__name__)

# test: test this module


class Camera:
    def __init__(self, intrinsic_matrix: np.array):
        self.K = intrinsic_matrix

    def get_k(self):
        return self.K

    def set_k(self, intrinsic_matrix):
        self.K = intrinsic_matrix


class Calibrator:

    @staticmethod
    def get_all_exif_params(image):
        """Return dict with image exif metadata.

        :param image: path to image on os
        :return: dict with exif parameters
        """

        logger.info("Getting image exif parameters;")
        exif = image.getexif()
        if not exif:
            return None

        exif_data = {
            PIL.ExifTags.TAGS[k]: v
            for k, v in exif.items()
            if k in PIL.ExifTags.TAGS
        }
        return exif_data

    @staticmethod
    def get_image_size(image: PIL.Image):
        """Return image size in pixels

        Args:
            image (PIL.Image): image object

        Returns:
            (tuple(int, int)): image width and height in pixels.
        """

        return image.size

    @staticmethod
    def get_sensor_size_by_camera_model(camera_model):
        return DataController.get_sensor_size_from_csv(camera_model)

    def get_params_from_exif_by_name(self, image, param_exif_names):
        result_dict = {}
        exif_params = self.get_all_exif_params(image)
        if not exif_params:
            logger.info("Image doesn't have exif parameters.")
            return result_dict
        for name in param_exif_names:
            result_dict[name] = exif_params[name] if name in exif_params else ''
        return result_dict

    def get_camera_params_from_exif(self, image):
        param_names = ['FocalLength', 'Model']
        params = self.get_params_from_exif_by_name(image, param_names)
        params["Image width"], params["Image height"] = self.get_image_size(image)
        if params['FocalLength']:
            params['FocalLength'] = float(params['FocalLength'])
        if params['Model']:
            params['Sensor width'], params['Sensor height'] = self.get_sensor_size_by_camera_model(params["Model"])
        return params

    @staticmethod
    def create_intrinsic_matrix(f_mm, image_width, image_height, sensor_width, sensor_height):
        """Return intrinsic camera matrix (K) by parameters

        K is
            | fx 0  u0 |
            | 0  fy v0 |
            | 0  0  1  |,

            where fx = width*f/(sensor width), fy = height*f/(sensor height)
                  u0, v0 is central point of the image frame (x and y, usually center of image).

        Args:
            f_mm (float): focal length in mm
            image_width (int): image width in pixels
            image_height (int): image height in pixels
            sensor_width (float): width of one pixel in world coordinates (mm)
            sensor_height (float): height of one pixel in world coordinates (mm)
        Returns:
            (np.array): intrinsic camera matrix (3x3)
        """

        fx = float(image_width*f_mm/sensor_width)
        fy = float(image_height*f_mm/sensor_height)
        return np.array([[fx, 0,  image_width/2],
                         [0,  fy, image_height/2],
                         [0,  0,  1]])
