from abc import ABC, abstractmethod
import cv2


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self):
        pass


class SIFT(FeatureExtractor):
    def __init__(self):
        self.sift = cv2.SIFT_create()

    def extract(self):
        pass

