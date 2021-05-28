import numpy as np
import cv2
from creator_3d.reconstuctor.actions.action import Reconstruct
from creator_3d.reconstuctor.constants import algorithm_default_params


class Reconstructor(Reconstruct):
    def __init__(self, action_name="Reconstruct", default_params_dict=None, **kwargs):
        if not default_params_dict:
            default_params_dict = algorithm_default_params.RECONSTRUCT_DEFAULT_PARAMS.copy()
        super(Reconstructor, self).__init__(action_name, default_params_dict, **kwargs)

        # params = self.__generate_params_dict(**kwargs)
        # self.flann = self.__get_flann_with_params(**params)

    def reset_params(self):
        pass

    @staticmethod
    def __get_projection(K, R, T):
        projection = np.zeros((3, 4))
        projection[0:3, 0:3] = np.float32(R)
        projection[:, 3] = np.float32(T.T)
        fk = np.float32(K)
        return np.dot(fk, projection)

    def reconstruct(self, **params):
        K = params.get("K")
        R1, T1 = params.get("R1"), params.get("T1")
        R2, T2 = params.get("R2"), params.get("T2")
        p1, p2 = params.get("p1"), params.get("p2")
        
        projection_1 = self.__get_projection(K, R1, T1)
        projection_2 = self.__get_projection(K, R2, T2)
        s = cv2.triangulatePoints(projection_1, projection_2, p1.T, p2.T)
        structure = []

        for i in range(len(s[0])):
            col = s[:, i]
            col /= col[3]
            structure.append([col[0], col[1], col[2]])

        return np.array(structure)
