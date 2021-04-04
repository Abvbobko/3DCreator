from creator_3d.reconstuctor.actions.action import Action
import PIL.Image
import PIL.ExifTags
import numpy as np


# todo: посмотреть, может можно получать image path из картинки opencv

class Calibrator(Action):
    def __init__(self):
        self.__action_name = "Calibrator"

    @staticmethod
    def get_exit_params(path_to_image):
        """Return dict with image exif metadata.

        :param path_to_image: path to image on os
        :return: dict with exif parameters
        """

        image = PIL.Image.open(path_to_image)
        exif_data = {
            PIL.ExifTags.TAGS[k]: v
            for k, v in image.getexif().items()
            if k in PIL.ExifTags.TAGS
        }
        return exif_data

    def __get__camera_intrinsic_matrix_from_exif(self, path_to_image):
        exif_params = self.get_exit_params(path_to_image)



    def run(self, **kwargs):
        pass

    @property
    def action_name(self):
        return self.__action_name

    # todo: чекать, есть ли EXIF данные
        # todo: если есть - достать калибровку (K) камеры
        # todo: если нету - откалибровать камеру

    # todo: можно сделать калибратор с клеточками (загружаешь фотку и оно калибрует по шахматам)
