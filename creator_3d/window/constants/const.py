import os

UI_FOLDER = './window'
UI_FILE_NAME = 'main_window.ui'
UI_PATH = os.path.join(UI_FOLDER, UI_FILE_NAME)

WIN_TITLE = "3DCreator"


class IMAGE_TABLE:
    table_header = ['Order', 'Name', 'Delete']

    @staticmethod
    def cols_count():
        return len(IMAGE_TABLE.table_header)


IMAGES_FOR_PROCESS_DIALOG_TITLE = "Open image files"
IMAGES_FOR_PROCESS_DIALOG_DIR = ".."
IMAGES_FOR_PROCESS_DIALOG_FILE_FILTER = "JPEG (*.jpg;*.jpeg);;PNG (*.png);;TIFF (*.tiff)"


# camera fields
class EditConstValue:
    def __init__(self, field_name=None, can_be_empty=False, max_length=255, mask_regex=None, placeholder=None):
        self.__field_name = field_name
        self.__can_be_empty = can_be_empty
        self.__max_length = max_length
        self.__mask_regex = mask_regex
        self.__placeholder = placeholder

    @property
    def field_name(self):
        return self.__field_name

    @property
    def can_be_empty(self):
        return self.__can_be_empty

    @property
    def max_length(self):
        return self.__max_length

    @property
    def mask_regex(self):
        return self.__mask_regex

    @property
    def placeholder(self):
        return self.__placeholder


FOCAL_LENGTH_EDIT_CONST = EditConstValue(field_name="focal length",
                                         max_length=9,
                                         can_be_empty=False,
                                         mask_regex=r"\d+(\.\d{1,4})?")

SENSOR_WIDTH_EDIT_CONST = EditConstValue(field_name="sensor width",
                                         max_length=5,
                                         can_be_empty=False,
                                         mask_regex=r"\d+(\.\d{1,2})?")

SENSOR_HEIGHT_EDIT_CONST = EditConstValue(field_name="sensor height",
                                          max_length=5,
                                          can_be_empty=False,
                                          mask_regex=r"\d+(\.\d{1,2})?")

IMAGE_WIDTH_EDIT_CONST = EditConstValue(field_name="image width",
                                        max_length=4,
                                        can_be_empty=False,
                                        mask_regex=r"\d+")

IMAGE_HEIGHT_EDIT_CONST = EditConstValue(field_name="image height",
                                         max_length=4,
                                         can_be_empty=False,
                                         mask_regex=r"\d+")

EXIF_IMAGE_DIALOG_TITLE = "Open image file"
EXIF_IMAGE_DIALOG_DIR = IMAGES_FOR_PROCESS_DIALOG_DIR
EXIF_IMAGE_DIALOG_FILE_FILTER = IMAGES_FOR_PROCESS_DIALOG_FILE_FILTER
