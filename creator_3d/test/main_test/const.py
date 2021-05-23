import creator_3d.test.intrinsic_parameters as intrinsic
import os

image_dir_path = 'C:\\Users\\hp\\Desktop\\3DCreator\\data'
dir_name = "rock_small"
image_dir = os.path.join(image_dir_path, dir_name)
MRT = 0.7

# K = intrinsic.MOBIUS

K = intrinsic.ROCK_K
# K = np.array([
#         [2362.12, 0, 720],
#         [0, 2362.12,  578],
#         [0, 0, 1]])

x = 0.5
y = 1
