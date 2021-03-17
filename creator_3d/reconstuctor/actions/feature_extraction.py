from abc import ABC, abstractmethod
from creator_3d.reconstuctor.actions.action import Action
import cv2


# по идее можно удалить
# class FeatureExtractor(Action, ABC):
#     @abstractmethod
#     def extract_features(self, image):
#         pass


class SIFT(Action):
    def __init__(self, action_name="SIFT"):
        self.sift = cv2.SIFT_create()
        self.__action_name = action_name

    def __extract_features(self, image):
        key_points, descriptors = self.sift.detectAndCompute(image, None)
        return key_points, descriptors

    # def get_result_dict(self):
    #     pass

    @property
    def action_name(self):
        return self.__action_name

    def run(self, **kwargs):
        image = kwargs['image']
        # todo: создавать словарь для следующего шага
        result_temp = self.__extract_features(image)
        return result_temp

