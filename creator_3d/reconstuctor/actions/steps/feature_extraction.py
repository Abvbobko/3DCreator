from creator_3d.reconstuctor.actions.action import Action
import cv2
import numpy as np


# по идее можно удалить
# class FeatureExtractor(Action, ABC):
#     @abstractmethod
#     def extract_features(self, image):
#         pass


class SIFT(Action):
    def __init__(self,
                 action_name="SIFT",
                 n_features=0,
                 n_octave_layers=3,
                 contrast_threshold=0.04,
                 edge_threshold=10,
                 sigma=1.6):

        self.sift = cv2.SIFT_create(n_features,
                                    n_octave_layers,
                                    contrast_threshold,
                                    edge_threshold,
                                    sigma)

        self.__action_name = action_name

    @property
    def action_name(self):
        return self.__action_name

    def run(self, **kwargs):
        image = kwargs['image']
        # todo: создавать словарь для следующего шага
        result_temp = self.__extract_features(image)
        return result_temp


class SURF(Action):
    pass


class ORB(Action):
    pass

