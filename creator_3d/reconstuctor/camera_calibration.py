import logging
import numpy as np
import PIL.ExifTags
import PIL.Image
from creator_3d.data_access.data_controller import DataController
from creator_3d.reconstuctor.constants import camera_const

logger = logging.getLogger(__name__)


class Camera:

    def __init__(self, focal_length, image_width, image_height, sensor_width, sensor_height, model=None):
        self.__model = model
        self.__focal_length = focal_length
        self.__image_width = image_width
        self.__image_height = image_height
        self.__sensor_width = sensor_width
        self.__sensor_height = sensor_height
        if self.__is_all_params_filled():
            self.__intrinsic_matrix = self.__create_intrinsic_matrix(f_mm=focal_length,
                                                                     image_width=image_width,
                                                                     image_height=image_height,
                                                                     sensor_width=sensor_width,
                                                                     sensor_height=sensor_height)
        else:
            self.__intrinsic_matrix = None

    @property
    def model_param_name(self):
        return camera_const.EXIF_PARAMS_NAME.model

    @property
    def focal_length_param_name(self):
        return camera_const.EXIF_PARAMS_NAME.focal_length

    @property
    def sensor_size_param_name(self):
        return camera_const.EXIF_PARAMS_NAME.sensor_width, camera_const.EXIF_PARAMS_NAME.sensor_height

    @property
    def image_size_param_name(self):
        return camera_const.EXIF_PARAMS_NAME.image_width, camera_const.EXIF_PARAMS_NAME.image_height

    @property
    def model(self):
        return self.__model

    @property
    def K(self):
        if self.__intrinsic_matrix is not None:
            return self.__intrinsic_matrix
        if self.__is_all_params_filled():
            return self.__create_intrinsic_matrix(f_mm=self.__focal_length,
                                                  image_width=self.__image_width,
                                                  image_height=self.__image_height,
                                                  sensor_width=self.__sensor_width,
                                                  sensor_height=self.__sensor_height)
        return None

    def __is_all_params_filled(self):
        return self.__focal_length and \
               self.__image_width and self.__image_height and \
               self.__sensor_width and self.__sensor_height

    @property
    def focal_length(self):
        return self.__focal_length

    @property
    def image_size(self):
        return self.__image_width, self.__image_height

    @property
    def sensor_size(self):
        return self.__sensor_width, self.__sensor_height

    @staticmethod
    def __create_intrinsic_matrix(f_mm, image_width, image_height, sensor_width, sensor_height):
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

        fx = float(image_width * f_mm / sensor_width)
        fy = float(image_height * f_mm / sensor_height)
        return np.array([[fx, 0, image_width / 2],
                         [0, fy, image_height / 2],
                         [0, 0, 1]])


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
        im_w_param_name = camera_const.EXIF_PARAMS_NAME.image_width
        im_h_param_name = camera_const.EXIF_PARAMS_NAME.image_height
        sw_param_name = camera_const.EXIF_PARAMS_NAME.sensor_width
        sh_param_name = camera_const.EXIF_PARAMS_NAME.sensor_height
        model_param_name = camera_const.EXIF_PARAMS_NAME.model
        focal_length_param_name = camera_const.EXIF_PARAMS_NAME.focal_length

        param_names = [focal_length_param_name, model_param_name]
        params = self.get_params_from_exif_by_name(image, param_names)

        params[im_w_param_name], params[im_h_param_name] = self.get_image_size(image)

        if focal_length_param_name in params and params[focal_length_param_name]:
            params[focal_length_param_name] = float(params[focal_length_param_name])
        else:
            params[focal_length_param_name] = ''
        if model_param_name in params and params[model_param_name]:
            params[sw_param_name], params[sh_param_name] = self.get_sensor_size_by_camera_model(
                params[model_param_name]
            )
        else:
            params[model_param_name] = None
            params[sw_param_name], params[sh_param_name] = '', ''

        return Camera(focal_length=params[focal_length_param_name],
                      image_width=params[im_h_param_name],
                      image_height=params[im_h_param_name],
                      sensor_width=params[sw_param_name],
                      sensor_height=params[sh_param_name],
                      model=params[model_param_name])

