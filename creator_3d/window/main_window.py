from PyQt5 import QtGui, uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QTableWidgetItem, QAbstractItemView
from PyQt5.QtCore import Qt

import os
import creator_3d.window.constants.const as window_const


class MainWindow(QMainWindow):
    def __init__(self, ui_path=window_const.UI_PATH):
        super(MainWindow, self).__init__()

        # init UI
        uic.loadUi(ui_path, self)

        # add drag and drop mode to image table
        self.image_table.setDragEnabled(True)
        self.image_table.setAcceptDrops(True)
        # self.image_table.viewport().setAcceptDrops(True)
        # self.image_table.setDragDropOverwriteMode(False)
        # self.image_table.setDropIndicatorShown(True)
        #
        # self.image_table.setSelectionMode(QAbstractItemView.SingleSelection)
        # self.image_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        # self.image_table.setDragDropMode(QAbstractItemView.InternalMove)

        # add some test data to table
        self.image_table.setColumnCount(1)
        self.image_table.setRowCount(3)
        test_text = ["text_1", "text_2", "text_3"]
        for i in range(len(test_text)):
            item = QTableWidgetItem(test_text[i])
            # item.setFlags(item.flags() | Qt.ItemIsSelectable)
            self.image_table.setItem(i, 0, item)

    # todo: Main
    # todo: add extension of existing widgets
    # todo: add parameters setting
    # todo: add algorithm choosing by combobox
    # todo: may be buttons add "Add images" and "Clear" instead of "Load images"
    # todo: add camera params input (f, sw, sh, get_camera_params)

    # todo: QTableWidget
    # todo: extend QTableWidget
    # todo: add drag and drop to QTableWidget
    # todo: add deleting from file list

    # todo: Load button
    # todo: show open file button
    # todo: choose files in dialog window
    # todo: add list of images to Table

    # todo: save full paths to images

    # todo: Process button
    # todo: process all images from list


