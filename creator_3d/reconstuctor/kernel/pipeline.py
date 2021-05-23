from typing import List

from creator_3d.reconstuctor.actions.action import Action
import cv2
import numpy as np


class Pipeline:
    def __init__(self, camera, feature_extractor):
        self.camera = camera
        self.feature_extractor = feature_extractor

    def extract_features(self, image_paths):
        # sift = cv2.SIFT_create(0, 3, 0.04, 10)
        key_points_for_all = []
        descriptor_for_all = []
        colors_for_all = []
        for image_path in image_paths:
            # todo: в дата контроллер
            image = cv2.imread(image_path)

            if image is None:
                continue

            # todo: заменить detectAndCompute
            key_points, descriptor = self.feature_extractor.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                                                             None)

            if len(key_points) <= 10:
                continue

            key_points_for_all.append(key_points)
            descriptor_for_all.append(descriptor)
            colors = np.zeros((len(key_points), 3))
            for i, key_point in enumerate(key_points):
                p = key_point.pt
                colors[i] = image[int(p[1])][int(p[0])]
            colors_for_all.append(colors)
        return np.array(key_points_for_all), np.array(descriptor_for_all), np.array(colors_for_all)

    def match_all_features(self, descriptors):
        matches_for_all = []
        for i in range(len(descriptors) - 1):
            matches = self.matcher.match_features(descriptors[i], descriptors[i + 1])
            matches_for_all.append(matches)
        return np.array(matches_for_all)

    def run(self, image_paths):

        k = self.camera.get_intrinsic_matrix()
        print("extract features")
        key_points_for_all, descriptor_for_all, colors_for_all = self.extract_features(image_paths)
        print("match features")
        matches_for_all = self.match_all_features(descriptor_for_all)
        print("reconstruct")

    # def __init__(self, steps: List[Action]):
    #     self.steps = steps
    #
    # def run(self, **kwargs):
    #     # start_params - dict with initial params
    #     # todo: можно вынести это в отдельный класс параметров, формировать его и получать нужное
    #     params = kwargs['start_params']
    #     for step in self.steps:
    #         step.run(**params)
    #         params = step.get_result_dict()
    #
    # def create_pipeline(self, steps: List[Action]):
    #     self.steps = steps



