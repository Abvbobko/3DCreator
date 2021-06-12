import numpy as np
import cv2
import creator_3d.reconstuctor.actions.action as actions
from creator_3d.reconstuctor.constants import algorithm_default_params as default_params


class Reconstructor(actions.Reconstruct):

    _default_params = default_params.RECONSTRUCT_DEFAULT_PARAMS.params
    _name = default_params.RECONSTRUCT_DEFAULT_PARAMS.name

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale = kwargs['scale']
        self.e_prob = kwargs['E_prob']
        self.e_threshold = kwargs['E_threshold']

    def __find_transform(self, K, p1, p2):
        """Find rotation and transform matrices"""

        focal_length = self.scale * (K[0, 0] + K[1, 1])
        principle_point = (K[0, 2], K[1, 2])
        E, mask = cv2.findEssentialMat(p1, p2, focal_length, principle_point, cv2.RANSAC,
                                       self.e_prob, self.e_threshold)
        camera_matrix = np.array(
            [[focal_length, 0, principle_point[0]], [0, focal_length, principle_point[1]], [0, 0, 1]])
        pass_count, R, T, mask = cv2.recoverPose(E, p1, p2, camera_matrix, mask)

        return R, T, mask

    @staticmethod
    def __get_projection(k, r, t):
        projection = np.zeros((3, 4))
        projection[0:3, 0:3] = np.float32(r)
        projection[:, 3] = np.float32(t.T)
        fk = np.float32(k)
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
