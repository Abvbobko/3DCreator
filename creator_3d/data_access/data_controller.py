# todo: module docstring
# todo: load image
    # cv
    # exif

import os
import logging

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
