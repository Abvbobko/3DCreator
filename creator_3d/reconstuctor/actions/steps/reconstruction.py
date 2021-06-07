import numpy as np
import cv2
import creator_3d.reconstuctor.actions.action as actions
from creator_3d.reconstuctor.constants import algorithm_default_params as default_params


class Reconstructor(actions.Reconstruct):

    __default_params = default_params.RECONSTRUCT_DEFAULT_PARAMS.params

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def __get_projection(k, r, t):
        projection = np.zeros((3, 4))
        projection[0:3, 0:3] = np.float32(r)
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

    def __str__(self):
        return default_params.RECONSTRUCT_DEFAULT_PARAMS.name
