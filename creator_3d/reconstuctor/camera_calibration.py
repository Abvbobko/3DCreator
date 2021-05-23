import logging
import numpy as np
import PIL.ExifTags
import PIL.Image

from creator_3d.reconstuctor.actions.action import Action

# database of sensor sizes
# https://github.com/openMVG/CameraSensorSizeDatabase

# todo: посмотреть, может можно получать image path из картинки open cv
# todo: можно сделать калибратор с клеточками (загружаешь фотку и оно калибрует по шахматам)

logger = logging.getLogger(__name__)

# constants
DEFAULT_SENSOR_WIDTH = 1
DEFAULT_SENSOR_HEIGHT = 1


class Calibrator:
    def __init__(self):
        self.__action_name = "Calibrator"

    @staticmethod
    def get_exif_params(image):
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
    def get_image_size(image_path):
        """Return image size in pixels

        Args:
            image_path (str): path to image

        Returns:
            (tuple(int, int)): image width and height in pixels.
            if image does not exist - return (None, None).
        """

        try:
            image = PIL.Image.open(image_path)
        except (FileNotFoundError, AttributeError):
            return None, None
        return image.size

    @staticmethod
    def read_image(image_path):
        try:
            image = PIL.Image.open(image_path)
        except FileNotFoundError:
            logger.error("Can't find image by path.")
            return None
        except AttributeError as e:
            logger.error("%s", e)
            # logger.error("Image doesn't contain exif parameters.")
            return None
        return image
    
    def get_camera_model_from_exif(self, image_path):
        logger.info("Getting camera model from image exif parameters (%s)",
                    image_path)
        image = image_path
        exif_params = self.get_exif_params(image)
        if not exif_params:
            logger.info("Image doesn't have exif parameters (%s)", image_path)
            return None
        camera_model = exif_params['Model']
        return camera_model

    def get_parameters_from_exif(self, image, names):
        result_dict = {}
        exif_params = self.get_exif_params(image)
        if not exif_params:
            logger.info("Image doesn't have exif parameters.")
            return result_dict
        for name in names:
            # todo: проверка, что содержит такой параметр exif_params
            result_dict[name] = exif_params[name]
        return result_dict

    def get_intrinsic_matrix_from_exif(self, image_path):
        # todo: нет смысла пока не научился получать sensor height/width
        logger.info("Getting camera intrinsic matrix (K) from image exif parameters (%s)",
                    image_path)

        image = self.read_image(image_path)
        if not image:
            return None

        # todo: всякие картинки на наличие словарей и тд
        focal_length = self.get_parameters_from_exif(image, ['FocalLength'])['FocalLength']
        width, height = image.size
        # todo: тут не так умножать надо скорее всего
        # todo: еще получить как то размер сенсора
        k = np.array([[float(focal_length*width), 0,                          width/2],
                      [0,                         float(focal_length*height), height/2],
                      [0,                         0,                          1]])
        return k

    def get_intrinsic_matrix(self, **kwargs):
        """Combine all provided parameters and
        return intrinsic matrix according this parameters.

        If camera parameters are provided - generate matrix
        if image_path - try to get camera parameters from exif

        Args:
            **kwargs:
                image_path (str): path to test image to get
                    intrinsic camera parameters (optional)
                f_mm (float): focal camera length in mm (optional)
                image_width (int): width of image in pixels (optional)
                image_height (int): height of image in pixels (optional)
                sensor_width (int): width of sensor in mm (optional, has default value)
                sensor_height (int): height of sensor in mm (optional, has default value)

        Returns:
            (np.array): intrinsic camera matrix (3x3)
        """

        # todo: тут много чего поменяется после добавления ui

        logger.info("Trying to get camera intrinsic matrix (K)")

        f_mm = kwargs.get('f_mm')

        im_width = kwargs.get('image_width')
        im_height = kwargs.get('image_height')
        sensor_width = kwargs.get('sensor_width', DEFAULT_SENSOR_WIDTH)
        sensor_height = kwargs.get('sensor_height', DEFAULT_SENSOR_HEIGHT)
        image_path = kwargs.get('image_path')

        k = None

        # if no image size - get size from test image
        if (im_width and im_height) is None:
            logger.info("There are no image width and height. Try to get them from test image (%s)",
                        image_path)
            im_width, im_height = self.get_image_size(image_path)

        # if we have all parameters - create K
        if (f_mm and im_width and im_height and sensor_width and sensor_height) is not None:
            k = self.create_intrinsic_matrix(f_mm=f_mm,
                                             image_width=im_width, image_height=im_height,
                                             sensor_width=sensor_width, sensor_height=sensor_height)
        elif image_path:
            # if we don't have some parameters - try to get K from test image
            k = self.get_intrinsic_matrix_from_exif(image_path)

        if k is None:
            # todo: calibrate
            logger.error("Couldn't get intrinsic camera matrix")
            raise Exception("Couldn't get intrinsic camera matrix")
    
        return k

    @staticmethod
    def create_intrinsic_matrix(f_mm, image_width, image_height, sensor_width=1, sensor_height=1):
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

    def run(self, **kwargs):

        logger.info("Start camera calibration")

        f_mm = kwargs.get('f_mm')
        image_width, image_height = kwargs.get('image_width'), kwargs.get('image_height')
        sensor_width, sensor_height = kwargs.get('sensor_width'), kwargs.get('sensor_height')
        image_path = kwargs.get('image_path')

        return self.get_intrinsic_matrix(f_mm=f_mm,
                                         image_width=image_width, image_height=image_height,
                                         sensor_width=sensor_width, sensor_height=sensor_height,
                                         image_path=image_path)

    @property
    def action_name(self):
        return self.__action_name


if __name__ == '__main__':
    print(Calibrator().get_intrinsic_matrix_from_exif(image_path='test/test.jpg'))
