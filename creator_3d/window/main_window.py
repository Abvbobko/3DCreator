from PyQt5 import QtGui, uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QTableWidgetItem, QAbstractItemView, QPushButton, \
    QHeaderView
from PyQt5.QtCore import Qt

import cv2
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
        self.image_table.setColumnCount(3)
        self.image_table.setRowCount(3)

        self.image_table.setHorizontalHeaderLabels(['Order', 'Name', 'Delete'])

        test_text = [[str(1), "0001.jpg", None],
                     [str(2), "0000.jpg", None],
                     [str(3), "0002.jpg", None]]

        header = self.image_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)

        for i in range(len(test_text)):
            for j in range(len(test_text[i])):
                if test_text[i][j]:
                    item = QTableWidgetItem(test_text[i][j])
                    self.image_table.setItem(i, j, item)
                else:
                    btn_sell = QPushButton('X')
                    # btn_sell.clicked.connect(self.handleButtonClicked)
                    self.image_table.setCellWidget(i, j, btn_sell)

        # for i in range(len(test_text)):
        #     item = QTableWidgetItem(test_text[i])
        #     # item.setFlags(item.flags() | Qt.ItemIsSelectable)
        #     self.image_table.setItem(i, 0, item)

        # test for 6 chapter
        self.feature_extraction_combobox.clear()
        self.feature_extraction_combobox.addItems(['SIFT', 'SURF', 'ORB'])
        self.feature_matching_combobox.clear()
        self.feature_matching_combobox.addItems(['FLANN', 'BF', 'KNN'])
        self.reconstruction_combobox.clear()
        self.reconstruction_combobox.addItems(['Reconstructor'])
        self.bundle_adjustment_combobox.clear()
        self.bundle_adjustment_combobox.addItems(['Bundle adjustment'])

        text = [["Feature extraction", "SIFT", "nfeatures", "nOctaveLayers", "contrastThreshold", "edgeThreshold",
                 "sigma"],
                ["", "SIFT", str(0), str(3), str(0.04), str(10), str(1.6)],
                ["Feature matching", "BF", "normType", "crossCheck", "MRT", "", ""],
                ["", "BF", "NORM_L2", "FALSE", "0.7", "", ""],
                ["Reconstruction", "Reconstructor", "E_prob", "E_threshold", "", "", ""],
                ["", "Reconstructor", "0.999", "1", "", "", ""],
                ["Bundle adjustment", "Bundle adjustment", "x_threshold", "y_threshold", "", "", ""],
                ["", "Bundle adjustment", "0.5", "1", "", "", ""]]
        self.algorithm_parameters_table.setColumnCount(len(text[0]))
        self.algorithm_parameters_table.setRowCount(len(text))
        self.algorithm_parameters_table.setHorizontalHeaderLabels(['Step', 'Algorithm', 'param_1', 'param_2', 'param_3',
                                                                   'param_4', 'param_5'])

        header = self.algorithm_parameters_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 7):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        for i in range(len(text)):
            for j in range(len(text[i])):
                item = QTableWidgetItem(text[i][j])
                self.algorithm_parameters_table.setItem(i, j, item)

        # test_text = ["text_1", "text_2", "text_3"]
        # for i in range(len(test_text)):
        #     # item.setFlags(item.flags() | Qt.ItemIsSelectable)
        #

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
    # todo: SORT TABLE AFTER CHANGE ORDER
    # Todo: все числа между старым и новым уменьшатся на 1



