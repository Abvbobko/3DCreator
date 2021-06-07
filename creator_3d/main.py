from PyQt5.QtWidgets import QApplication
from creator_3d.window.main_window import MainWindow
import sys


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == '__main__':

    sys.excepthook = except_hook
    app = QApplication([])
    window = MainWindow()
    window.show()
    # Start the event loop.
    app.exec_()
