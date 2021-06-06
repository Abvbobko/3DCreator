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
