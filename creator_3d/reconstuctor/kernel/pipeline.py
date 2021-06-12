import cv2
import numpy as np
import math
from creator_3d.reconstuctor.constants import pipeline_const
import logging
from creator_3d.reconstuctor.actions.action import Extract, Match, Reconstruct, BundleAdjustment
from creator_3d.reconstuctor.camera_calibration import Camera

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self,
                 camera,
                 feature_extractor: Extract,
                 feature_matcher: Match,
                 reconstructor: Reconstruct,
                 bundle_adjuster: BundleAdjustment):
        self.camera = camera
        self.extractor = feature_extractor
        self.matcher = feature_matcher
        self.reconstructor = reconstructor
        self.bundle_adjuster = bundle_adjuster

    def __extract_features(self, image_names):
        """Extract all key points and descriptors from image list.

        Args:
            image_names (list): list of images
        Returns:
            (np.array): matrix with key points
            (np.array): matrix with descriptors
        """

        logger.info("Extract features")
        key_points_for_all = []
        descriptor_for_all = []
        image_number = 0
        for image_name in image_names:
            logger.info("Extract from image #%s", image_number)
            image_number += 1

            # todo: move
            image = cv2.imread(image_name)

            if image is None:
                continue

            key_points, descriptor = self.extractor.detect_and_compute(image)

            if len(key_points) <= pipeline_const.MIN_NUMBER_OF_KEY_POINTS:
                continue

            key_points_for_all.append(key_points)
            descriptor_for_all.append(descriptor)
        logger.info("Extraction is finished.")
        return np.array(key_points_for_all), np.array(descriptor_for_all)

    def __match_features(self, descriptors):
        """Match features between images"""
        logger.info("Match features.")
        matches_for_all_images = []
        for i in range(len(descriptors) - 1):
            logger.info("Match features for images %s - %s", i, i + 1)
            matches = self.matcher.match_features(descriptors[i], descriptors[i + 1])
            matches_for_all_images.append(matches)
        logger.info("Matching is finished.")
        return np.array(matches_for_all_images)

    @staticmethod
    def __get_matched_points(p1, p2, matches):
        src_pts = np.asarray([p1[m.queryIdx].pt for m in matches])
        dst_pts = np.asarray([p2[m.trainIdx].pt for m in matches])

        return src_pts, dst_pts

    @staticmethod
    def __mask_out_points(p1, mask):
        p1_copy = []
        for i in range(len(mask)):
            if mask[i] > 0:
                p1_copy.append(p1[i])

        return np.array(p1_copy)

    def __init_structure(self, key_points_for_all, matches_for_all):
        """Create structure with the first items"""
        p1, p2 = self.__get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0])

        K = self.camera.K
        transform = self.reconstructor.__find_transform(K, p1, p2)
        if transform:
            R, T, mask = transform
        else:
            R, T, mask = np.array([]), np.array([]), np.array([])

        p1 = self.__mask_out_points(p1, mask)
        p2 = self.__mask_out_points(p2, mask)
        R0 = np.eye(3, 3)
        T0 = np.zeros((3, 1))
        structure = self.reconstructor.reconstruct(K=K, R1=R0, T1=T0, R2=R, T2=T, p1=p1, p2=p2)
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

    def run(self, image_names):
        """Run image processing

        Args:
            image_names (list[str]): list of image paths

        Returns:
            (np.array): result point cloud.
        """

        # todo: think how to implement image loading
        key_points, descriptor = self.__extract_features(image_names)
        matches = self.__match_features(descriptor)
        logger.info("Reconstruction")
        print("0 - 1")
        structure, correspond_struct_idx, rotations, motions = self.__init_structure(key_points,
                                                                                     matches)

        for i in range(1, len(matches)):
            print(f"{i} - {i + 1}")
            object_points, image_points = self.__get_objpoints_and_imgpoints(matches[i],
                                                                             correspond_struct_idx[i],
                                                                             structure,
                                                                             key_points[i + 1])

            if len(image_points) < 7:
                while len(image_points) < 7:
                    object_points = np.append(object_points, [object_points[0]], axis=0)
                    image_points = np.append(image_points, [image_points[0]], axis=0)

            _, r, T, _ = cv2.solvePnPRansac(object_points, image_points, self.camera.K, np.array([]))
            R, _ = cv2.Rodrigues(r)
            rotations.append(R)
            motions.append(T)
            p1, p2 = self.__get_matched_points(key_points[i],
                                               key_points[i + 1],
                                               matches[i])
            next_structure = self.reconstructor.reconstruct(K=self.camera.K,
                                                            R1=rotations[i],
                                                            T1=motions[i],
                                                            R2=R,
                                                            T2=T,
                                                            p1=p1,
                                                            p2=p2)

            correspond_struct_idx[i], correspond_struct_idx[i + 1], structure = self.__fusion_structure(
                matches[i],
                correspond_struct_idx[i],
                correspond_struct_idx[i + 1],
                structure, next_structure
            )

        structure = self.bundle_adjuster.adjust_bundle(rotations=rotations,
                                                       motions=motions,
                                                       K=self.camera.K,
                                                       correspond_struct_idx=correspond_struct_idx,
                                                       key_points=key_points,
                                                       structure=structure)
        i = 0
        while i < len(structure):
            if math.isnan(structure[i][0]):
                structure = np.delete(structure, i, 0)
                i -= 1
            i += 1

        return structure
