from PyQt5 import QtGui, uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QTableWidgetItem, QAbstractItemView

import os
import creator_3d.window.constants.const as window_const


class MainWindow(QMainWindow):
    def __init__(self, ui_path=window_const.UI_PATH):
        super(MainWindow, self).__init__()

        # init UI
        uic.loadUi(ui_path, self)

