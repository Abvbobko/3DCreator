from abc import ABC, abstractmethod
import cv2


class FeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, image):
        pass


class SIFT(FeatureExtractor):
    def __init__(self):
        self.sift = cv2.SIFT_create()

    def extract_features(self, image):
        key_points, descriptors = self.sift.detectAndCompute(image, None)
        return key_points, descriptors


