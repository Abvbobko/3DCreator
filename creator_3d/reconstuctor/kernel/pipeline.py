from typing import List

from creator_3d.reconstuctor.actions.action import Action
import cv2
import numpy as np
from creator_3d.data_access.data_controller import DataController
import math


class Pipeline:
    def __init__(self, camera, feature_extractor):
        self.camera = camera
        self.feature_extractor = feature_extractor
        self.extractor = None
        self.matcher = None
        # todo: may be set K in init?

    def __extract_features(self, image_names):
        key_points_for_all = []
        descriptor_for_all = []
        for image_name in image_names:
            image = cv2.imread(image_name)

            if image is None:
                continue

            key_points, descriptor = self.extractor.detect_and_compute(image, None)

            # todo: const
            if len(key_points) <= 10:
                continue

            key_points_for_all.append(key_points)
            descriptor_for_all.append(descriptor)
            # # todo: remove work with colors
            # colors = np.zeros((len(key_points), 3))
            # for i, key_point in enumerate(key_points):
            #     p = key_point.pt
            #     colors[i] = image[int(p[1])][int(p[0])]
            # colors_for_all.append(colors)
        return np.array(key_points_for_all), np.array(descriptor_for_all) # , np.array(colors_for_all)

    # todo: to matcher
    # def match_features(self, query, train):
    #     bf = cv2.BFMatcher(cv2.NORM_L2)
    #     knn_matches = bf.knnMatch(query, train, k=2)
    #     matches = []
    #     for m, n in knn_matches:
    #         if m.distance < const.MRT * n.distance:
    #             matches.append(m)
    #
    #     return np.array(matches)

    def __match_features(self, descriptor_for_all):
        matches_for_all = []
        for i in range(len(descriptor_for_all) - 1):
            matches = self.matcher.match_features(descriptor_for_all[i], descriptor_for_all[i + 1])
            matches_for_all.append(matches)
        return np.array(matches_for_all)

    @staticmethod
    def __get_matched_points(p1, p2, matches):
        src_pts = np.asarray([p1[m.queryIdx].pt for m in matches])
        dst_pts = np.asarray([p2[m.trainIdx].pt for m in matches])

        return src_pts, dst_pts

    @staticmethod
    def __find_transform(K, p1, p2):
        focal_length = 0.5 * (K[0, 0] + K[1, 1]) # todo: 0.5 to const?
        principle_point = (K[0, 2], K[1, 2])
        E, mask = cv2.findEssentialMat(p1, p2, focal_length, principle_point, cv2.RANSAC, 0.999, 1.0)  # it's params !!!
        camera_matrix = np.array(
            [[focal_length, 0, principle_point[0]], [0, focal_length, principle_point[1]], [0, 0, 1]])
        pass_count, R, T, mask = cv2.recoverPose(E, p1, p2, camera_matrix, mask)

        return R, T, mask

    @staticmethod
    def __maskout_points(p1, mask):
        p1_copy = []
        for i in range(len(mask)):
            if mask[i] > 0:
                p1_copy.append(p1[i])

        return np.array(p1_copy)

    @staticmethod
    def __reconstruct(K, R1, T1, R2, T2, p1, p2):
        # todo reconstructor
        proj1 = np.zeros((3, 4))
        proj2 = np.zeros((3, 4))
        proj1[0:3, 0:3] = np.float32(R1)
        proj1[:, 3] = np.float32(T1.T)
        proj2[0:3, 0:3] = np.float32(R2)
        proj2[:, 3] = np.float32(T2.T)
        fk = np.float32(K)
        proj1 = np.dot(fk, proj1)
        proj2 = np.dot(fk, proj2)
        s = cv2.triangulatePoints(proj1, proj2, p1.T, p2.T)
        structure = []

        for i in range(len(s[0])):
            col = s[:, i]
            col /= col[3]
            structure.append([col[0], col[1], col[2]])

        return np.array(structure)

    def __init_structure(self, K, key_points_for_all, matches_for_all):
        # todo: rename
        p1, p2 = self.__get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0])
        # c1, c2 = self.get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0])

        if self.__find_transform(K, p1, p2):
            R, T, mask = self.__find_transform(K, p1, p2)
        else:
            R, T, mask = np.array([]), np.array([]), np.array([])

        p1 = self.__maskout_points(p1, mask)
        p2 = self.__maskout_points(p2, mask)
        # colors = self.__maskout_points(c1, mask)
        R0 = np.eye(3, 3)
        T0 = np.zeros((3, 1))
        structure = self.__reconstruct(K, R0, T0, R, T, p1, p2)
        rotations = [R0, R]
        motions = [T0, T]
        correspond_struct_idx = []
        for key_p in key_points_for_all:
            correspond_struct_idx.append(np.ones(len(key_p)) * - 1)
        correspond_struct_idx = np.array(correspond_struct_idx)
        idx = 0
        matches = matches_for_all[0]
        for i, match in enumerate(matches):
            if mask[i] == 0:
                continue
            correspond_struct_idx[0][int(match.queryIdx)] = idx
            correspond_struct_idx[1][int(match.trainIdx)] = idx
            idx += 1
        return structure, correspond_struct_idx, rotations, motions

    @staticmethod
    def __get_objpoints_and_imgpoints(matches, struct_indices, structure, key_points):
        """Creating image points and spatial points"""

        object_points = []
        image_points = []
        for match in matches:
            query_idx = match.queryIdx
            train_idx = match.trainIdx
            struct_idx = struct_indices[query_idx]
            if struct_idx < 0:
                continue
            object_points.append(structure[int(struct_idx)])
            image_points.append(key_points[train_idx].pt)

        return np.array(object_points), np.array(image_points)

    @staticmethod
    def __fusion_structure(matches, struct_indices, next_struct_indices, structure, next_structure):
        """Point cloud convergence that was made"""

        for i, match in enumerate(matches):
            query_idx = match.queryIdx
            train_idx = match.trainIdx
            struct_idx = struct_indices[query_idx]
            if struct_idx >= 0:
                next_struct_indices[train_idx] = struct_idx
                continue
            # todo: we can change append to list and it will faster
            structure = np.append(structure, [next_structure[i]], axis=0)
            struct_indices[query_idx] = next_struct_indices[train_idx] = len(structure) - 1
        return struct_indices, next_struct_indices, structure

    def get_3dpos_v1(self, pos, ob, r, t, K):
        p, J = cv2.projectPoints(pos.reshape(1, 1, 3), r, t, K, np.array([]))
        p = p.reshape(2)
        e = ob - p
        if abs(e[0]) > const.x or abs(e[1]) > const.y:
            return None
        return pos

    @staticmethod
    def __bundle_adjustment(rotations, motions, K, correspond_struct_idx, key_points_for_all, structure):
        # todo to step bundle
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
                new_point = get_3dpos_v1(structure[point3d_id], key_points[j].pt, r, t, K)
                structure[point3d_id] = new_point

        return structure

    def run(self, image_names, K):
        print("extract features")
        # todo: change on log
        # todo: think how to implement image loading
        key_points_for_all, descriptor_for_all = self.__extract_features(image_names)
        print("match features")
        matches_for_all = self.__match_features(descriptor_for_all)
        print("reconstruct")
        print("0 - 1")
        structure, correspond_struct_idx, rotations, motions = self.__init_structure(K,
                                                                                     key_points_for_all,
                                                                                     matches_for_all)

        for i in range(1, len(matches_for_all)):
            print(f"{i} - {i + 1}")
            object_points, image_points = self.__get_objpoints_and_imgpoints(matches_for_all[i],
                                                                             correspond_struct_idx[i],
                                                                             structure,
                                                                             key_points_for_all[i + 1])
            # In python's opencv, the first parameter length of the solvePnPRansac
            # function needs to be greater than 7,otherwise an error will be reported
            # Here do a repeat fill operation for the set of points less than 7,
            # that is, fill 7 with the first point in the set of points
            if len(image_points) < 7:
                while len(image_points) < 7:
                    object_points = np.append(object_points, [object_points[0]], axis=0)
                    image_points = np.append(image_points, [image_points[0]], axis=0)

            _, r, T, _ = cv2.solvePnPRansac(object_points, image_points, K, np.array([]))
            R, _ = cv2.Rodrigues(r)
            rotations.append(R)
            motions.append(T)
            p1, p2 = self.__get_matched_points(key_points_for_all[i],
                                               key_points_for_all[i + 1],
                                               matches_for_all[i])
            next_structure = self.__reconstruct(K, rotations[i], motions[i], R, T, p1, p2)

            correspond_struct_idx[i], correspond_struct_idx[i + 1], structure = self.__fusion_structure(
                matches_for_all[i],
                correspond_struct_idx[i],
                correspond_struct_idx[i + 1],
                structure, next_structure
            )

        structure = self.__bundle_adjustment(rotations,
                                             motions,
                                             K,
                                             correspond_struct_idx,
                                             key_points_for_all,
                                             structure)
        i = 0
        # Due to the bundle_adjustment structure, some empty points are
        # generated(the actual representative means they were removed）
        # Remove these empty dots here
        while i < len(structure):
            if math.isnan(structure[i][0]):
                structure = np.delete(structure, i, 0)
                # colors = np.delete(colors, i, 0)
                i -= 1
            i += 1

        # todo: save model to file
        fig_v1(structure)

    # def extract_features(self, image_paths):
    #     # sift = cv2.SIFT_create(0, 3, 0.04, 10)
    #     key_points_for_all = []
    #     descriptor_for_all = []
    #     colors_for_all = []
    #     for image_path in image_paths:
    #         # todo: в дата контроллер
    #         image = cv2.imread(image_path)
    #
    #         if image is None:
    #             continue
    #
    #         # todo: заменить detectAndCompute
    #         key_points, descriptor = self.feature_extractor.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
    #                                                                          None)
    #
    #         if len(key_points) <= 10:
    #             continue
    #
    #         key_points_for_all.append(key_points)
    #         descriptor_for_all.append(descriptor)
    #         colors = np.zeros((len(key_points), 3))
    #         for i, key_point in enumerate(key_points):
    #             p = key_point.pt
    #             colors[i] = image[int(p[1])][int(p[0])]
    #         colors_for_all.append(colors)
    #     return np.array(key_points_for_all), np.array(descriptor_for_all), np.array(colors_for_all)
    #
    # def match_all_features(self, descriptors):
    #     matches_for_all = []
    #     for i in range(len(descriptors) - 1):
    #         matches = self.matcher.match_features(descriptors[i], descriptors[i + 1])
    #         matches_for_all.append(matches)
    #     return np.array(matches_for_all)
    #
    # def run(self, image_paths):
    #
    #     k = self.camera.get_intrinsic_matrix()
    #     print("extract features")
    #     key_points_for_all, descriptor_for_all, colors_for_all = self.extract_features(image_paths)
    #     print("match features")
    #     matches_for_all = self.match_all_features(descriptor_for_all)
    #     print("reconstruct")
    #
    # # def __init__(self, steps: List[Action]):
    # #     self.steps = steps
    # #
    # # def run(self, **kwargs):
    # #     # start_params - dict with initial params
    # #     # todo: можно вынести это в отдельный класс параметров, формировать его и получать нужное
    # #     params = kwargs['start_params']
    # #     for step in self.steps:
    # #         step.run(**params)
    # #         params = step.get_result_dict()
    # #
    # # def create_pipeline(self, steps: List[Action]):
    # #     self.steps = steps
    #
    #
    #
