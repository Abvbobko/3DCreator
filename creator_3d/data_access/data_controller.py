# todo: module docstring
# todo: load image
    # cv
    # exif

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


class DataController:
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
        with open(txt_file) as f:
            lines = f.readlines()
        return np.array(
            [l.strip().split(' ') for l in lines],
            dtype=np.float32
        )

    @staticmethod
    def safe_mkdir(file_dir):
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)

    @staticmethod
    def write_simple_obj(mesh_v, mesh_f, file_path):
        with open(file_path, 'w') as fp:
            for v in mesh_v:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            if mesh_f is not None:
                for f in mesh_f + 1:
                    fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
