# todo: module docstring
# todo: get sensor params from file using camera model

import os
import logging
import numpy as np
import PIL.Image
import cv2
import csv
import creator_3d.data_access.constants.const as const

logger = logging.getLogger(__name__)


class DataController:

    @staticmethod
    def get_full_image_paths(image_dir, image_names):
        return [os.path.join(image_dir, image_names)]

    @staticmethod
    def get_dir_content(path_to_dir):
        """Get list of dir content.
        If dir does not exist - return empty list.

        Args:
            path_to_dir (str): path to folder on system.
        Returns:
            (list[str]): list of dir content
        """

        try:
            content_list = os.listdir(path_to_dir)
        except FileNotFoundError:
            logger.error(f"Directory %s does not exist.", path_to_dir)
            content_list = []
        return content_list

    @staticmethod
    def intrinsic_reader(txt_file):
        """Read intrinsic camera matrix from file.

        Args:
            txt_file (str): path to file with matrix inside.

        Returns:
            (np.array): camera intrinsic matrix.
        """

        with open(txt_file) as f:
            lines = f.readlines()
        return np.array(
            [l.strip().split(' ') for l in lines],
            dtype=np.float32
        )

    @staticmethod
    def safe_mkdir(file_dir):
        """Create dir if it does not exist"""

        if not os.path.exists(file_dir):
            os.mkdir(file_dir)

    @staticmethod
    def write_obj_file(mesh_v, mesh_f, file_path):
        """Write point cloud to .obj file

        Args:
            mesh_v (np.array): coordinates of points
            mesh_f (np.array): faces of model
            file_path (str): path to result file
        """

        with open(file_path, 'w') as obj_file:
            for v in mesh_v:
                obj_file.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            if mesh_f is not None:
                for f in mesh_f + 1:
                    obj_file.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    @staticmethod
    def write_ply_file(mesh_v, file_path):
        """Write point cloud to .ply file

        Args:
            mesh_v (np.array): coordinates of points
            file_path (str): path to result file
        """

        with open(file_path, 'w') as ply_file:
            # Write header of .ply file
            ply_file.write('ply\n')
            ply_file.write('format ascii 1.0\n')
            ply_file.write('element vertex %d\n' % mesh_v.shape[0])
            ply_file.write('property float x\n')
            ply_file.write('property float y\n')
            ply_file.write('property float z\n')
            ply_file.write('end_header\n')

            for v in mesh_v:
                ply_file.write('%f %f %f\n' % (v[0], v[1], v[2]))

    @staticmethod
    def read_pil_image(image_path):
        """Read image using PIL lib"""

        try:
            image = PIL.Image.open(image_path)
        except FileNotFoundError:
            logger.error("Can't find image by path.")
            return None
        except AttributeError as e:
            logger.error("%s", e)
            return None
        return image

    @staticmethod
    def read_cv2_image(image_path):
        """Read image using cv2 lib (Mat object)"""

        try:
            image = cv2.imread(image_path)
        except FileNotFoundError:
            logger.error("Can't find image by path.")
            return None
        except AttributeError as e:
            logger.error("%s", e)
            return None
        return image

    @staticmethod
    def get_sensor_size_from_csv(camera_model):
        """Tuple (sensor width, sensor height) in mm"""

        camera_make_and_model = camera_model.split(" ", 1)
        model = camera_make_and_model[0] if len(camera_make_and_model) == 1 else camera_make_and_model[1]

        with open(const.SENSOR_DATABASE_PATH, 'r') as sensor_database_csv:
            reader = csv.DictReader(sensor_database_csv)

            for row in reader:
                if row[const.CsvHeaderParamName.camera_model].upper() in [model.upper(), camera_model.upper()]:
                    return float(row[const.CsvHeaderParamName.sensor_width]), \
                           float(row[const.CsvHeaderParamName.sensor_height])

        return '', ''
