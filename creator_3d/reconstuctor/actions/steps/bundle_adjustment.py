import creator_3d.reconstuctor.actions.action as actions
from creator_3d.reconstuctor.constants import algorithm_default_params as default_params
from creator_3d.reconstuctor.constants import pipeline_const
import cv2
import numpy as np


class BundleAdjuster(actions.BundleAdjustment):

    _default_params = default_params.BUNDLE_ADJUSTMENT_DEFAULT_PARAMS.params
    _name = default_params.BUNDLE_ADJUSTMENT_DEFAULT_PARAMS.name

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.x_threshold = kwargs.get('x_threshold')
        self.y_threshold = kwargs.get('y_threshold')

    def __get_3d_pos(self, pos, ob, r, t, K):
        p, J = cv2.projectPoints(pos.reshape(1, 1, 3), r, t, K, np.array([]))
        p = p.reshape(2)
        e = ob - p
        if abs(e[0]) > self.x_threshold or abs(e[1]) > self.y_threshold:
            return None
        return pos

    def adjust_bundle(self, **params):
        rotations = params.get("rotations")
        motions = params.get("motions")
        K = params.get("K")
        correspond_struct_idx = params.get("correspond_struct_idx")
        key_points_for_all = params.get("key_points")
        structure = params.get("structure")

        for i in range(len(rotations)):
            r, _ = cv2.Rodrigues(rotations[i])
            rotations[i] = r
        for i in range(len(correspond_struct_idx)):
            point3d_ids = correspond_struct_idx[i]
            key_points = key_points_for_all[i]
            r = rotations[i]
            t = motions[i]
            for j in range(len(point3d_ids)):
                point3d_id = int(point3d_ids[j])
                if point3d_id < 0:
                    continue
                new_point = self.__get_3d_pos(structure[point3d_id], key_points[j].pt, r, t, K)
                structure[point3d_id] = new_point

        return structure
