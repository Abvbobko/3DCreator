from PyQt5 import QtGui, uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QTableWidgetItem, QAbstractItemView

import os


class MainWindow(QMainWindow):
    def __init__(self, ui_path):
        super(MainWindow, self).__init__()

        # init UI
        uic.loadUi(ui_path, self)

