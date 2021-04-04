from creator_3d.reconstuctor.actions.action import Action
import PIL.Image
import PIL.ExifTags
import numpy as np


# todo: посмотреть, может можно получать image path из картинки opencv

class Calibrator(Action):
    def __init__(self):
        self.__action_name = "Calibrator"

    @staticmethod
    def get_exit_params(image):
        """Return dict with image exif metadata.

        :param image: path to image on os
        :return: dict with exif parameters
        """

        exif = image.getexif()
        if not exif:
            return None

        exif_data = {
            PIL.ExifTags.TAGS[k]: v
            for k, v in exif.items()
            if k in PIL.ExifTags.TAGS
        }
        return exif_data

    def __get__camera_intrinsic_matrix_from_exif(self, image_path):
        image = PIL.Image.open(image_path)
        exif_params = self.get_exit_params(image)
        if not exif_params:
            return None
        focal_length = exif_params['FocalLength']
        width, height = image.size
        k = np.array([[focal_length*width, 0,                   width/2],
                      [0,                  focal_length*height, height/2],
                      [0,                  0,                   1]])
        return k

    def __calibrate(self, image_path):
        k = self.__get__camera_intrinsic_matrix_from_exif(image_path)
        if k is not None:
            return k

        # todo: call __calibrate_with_F_matrix?

    def run(self, **kwargs):
        image_path = kwargs['image_1_path']
        return self.__calibrate(image_path)

    @property
    def action_name(self):
        return self.__action_name

    # todo: чекать, есть ли EXIF данные
        # todo: если есть - достать калибровку (K) камеры
        # todo: если нету - откалибровать камеру

    # todo: можно сделать калибратор с клеточками (загружаешь фотку и оно калибрует по шахматам)
