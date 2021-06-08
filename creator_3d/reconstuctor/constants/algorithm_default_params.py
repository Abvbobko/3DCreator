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
SIFT_DEFAULT_PARAMS = AlgorithmDefaultParams("SIFT", {"n_features": 0,
                                                      "n_octave_layers": 3,
                                                      "contrast_threshold": 0.04,
                                                      "edge_threshold": 10,
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
RECONSTRUCT_DEFAULT_PARAMS = AlgorithmDefaultParams("Reconstruct", {})

# bundle adjustment
BUNDLE_ADJUSTMENT_DEFAULT_PARAMS = AlgorithmDefaultParams("Bundle adjust", {})
