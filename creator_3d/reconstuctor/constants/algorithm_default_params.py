import cv2

# extraction
SIFT_DEFAULT_PARAMS = {
    "n_features": 0,
    "n_octave_layers": 3,
    "contrast_threshold": 0.04,
    "edge_threshold": 10,
    "sigma": 1.6
}

# matching
FLANN_DEFAULT_PARAMS = {
    "index_kdtree": 1,
    "trees": 5,
    "checks": 50
}

# matching
BF_DEFAULT_PARAMS = {
    "normType": cv2.NORM_L2,
    "crossCheck": False
}


# reconstruction
RECONSTRUCT_DEFAULT_PARAMS = {}

BUNDLE_ADJUSTMENT_DEFAULT_PARAMS = {}
