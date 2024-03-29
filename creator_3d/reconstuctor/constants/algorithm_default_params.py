import cv2


class AlgorithmDefaultParams:
    def __init__(self, algorithm_name, params_dict):
        self.__params = params_dict
        self.__name = algorithm_name

    @property
    def params(self):
        return self.__params

    @property
    def name(self):
        return self.__name


# extraction
SIFT_DEFAULT_PARAMS = AlgorithmDefaultParams("SIFT", {"nfeatures": 0,
                                                      "nOctaveLayers": 3,
                                                      "contrastThreshold": 0.04,
                                                      "edgeThreshold": 10,
                                                      "sigma": 1.6})

SURF_DEFAULT_PARAMS = AlgorithmDefaultParams("SURF", {"hessianThreshold": 100,
                                                      "nOctaves": 4,
                                                      "nOctaveLayers": 3,
                                                      "extended": False,
                                                      "upright": False})

ORB_DEFAULT_PARAMS = AlgorithmDefaultParams("ORB", {"nfeatures": 500,
                                                    "scaleFactor": 1.2,
                                                    "nlevels": 8,
                                                    "edgeThreshold": 31,
                                                    "firstLevel": 0,
                                                    "WTA_K": 2,
                                                    "scoreType": cv2.ORB_HARRIS_SCORE,
                                                    "patchSize": 31,
                                                    "fastThreshold": 20})

# matching
FLANN_DEFAULT_PARAMS = AlgorithmDefaultParams("FLANNMatcher", {"index_kdtree": 1,
                                                               "trees": 5,
                                                               "checks": 50})

BF_DEFAULT_PARAMS = AlgorithmDefaultParams("BFMatcher", {"normType": cv2.NORM_L2,
                                                         "crossCheck": False})


# reconstruction
RECONSTRUCT_DEFAULT_PARAMS = AlgorithmDefaultParams("Reconstructor", {'scale': 0.5,
                                                                      'E_prob': 0.999,
                                                                      'E_threshold': 1.0})

# bundle adjustment
BUNDLE_ADJUSTMENT_DEFAULT_PARAMS = AlgorithmDefaultParams("BundleAdjuster", {'x_threshold': 0,
                                                                             'y_threshold': 1})
