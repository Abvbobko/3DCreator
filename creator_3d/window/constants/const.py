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
