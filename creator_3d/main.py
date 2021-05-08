from PyQt5.QtWidgets import QApplication
from creator_3d.window.main_window import MainWindow


if __name__ == '__main__':

    app = QApplication([])
    window = MainWindow()
    window.show()
    # Start the event loop.
    app.exec_()
