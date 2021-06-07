from PyQt5 import QtGui, uic
from PyQt5.QtWidgets import (QApplication,
                             QMainWindow,
                             QMessageBox,
                             QTableWidgetItem,
                             QAbstractItemView,
                             QPushButton,
                             QHeaderView,
                             QFileDialog,
                             qApp)
from PyQt5.QtCore import Qt

import cv2
import os
import creator_3d.window.constants.const as window_const
from creator_3d.controllers.main_controller import MainController


class MainWindow(QMainWindow):
    def __init__(self, ui_path=window_const.UI_PATH):
        self.main_controller = MainController()

        super().__init__()

        # init UI
        uic.loadUi(ui_path, self)

        self.path_to_image_files = None

        # set window title
        self.setWindowTitle(window_const.WIN_TITLE)

        # image loading
        self.load_images_button.clicked.connect(self.__choose_files_and_fill_image_table)

        self.image_table.setColumnCount(window_const.IMAGE_TABLE.cols_count())
        self.image_table.setHorizontalHeaderLabels(window_const.IMAGE_TABLE.table_header)

        # set width of column using size of text in header
        header = self.image_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)

        # combobox filling
        self.feature_extraction_combobox.step_name = self.main_controller.get_extract_step_name()
        self.feature_extraction_combobox.clear()
        self.feature_extraction_combobox.addItems(
            self.main_controller.get_step_algorithms(self.feature_extraction_combobox.step_name)
        )

        self.feature_matching_combobox.step_name = self.main_controller.get_match_step_name()
        self.feature_matching_combobox.clear()
        self.feature_matching_combobox.addItems(
            self.main_controller.get_step_algorithms(self.feature_matching_combobox.step_name)
        )

        self.reconstruction_combobox.step_name = self.main_controller.get_reconstruct_step_name()
        self.reconstruction_combobox.clear()
        self.reconstruction_combobox.addItems(
            self.main_controller.get_step_algorithms(self.reconstruction_combobox.step_name)
        )

        self.bundle_adjustment_combobox.step_name = self.main_controller.get_bundle_adjust_step_name()
        self.bundle_adjustment_combobox.clear()
        self.bundle_adjustment_combobox.addItems(
            self.main_controller.get_step_algorithms(self.bundle_adjustment_combobox.step_name)
        )

        # algo params filling todo remove
        # text = [["Feature extraction", "SIFT", "nfeatures", "nOctaveLayers", "contrastThreshold", "edgeThreshold",
        #          "sigma"],
        #         ["", "SIFT", str(0), str(3), str(0.04), str(10), str(1.6)],
        #         ["Feature matching", "BF", "normType", "crossCheck", "MRT", "", ""],
        #         ["", "BF", "NORM_L2", "FALSE", "0.7", "", ""],
        #         ["Reconstruction", "Reconstructor", "E_prob", "E_threshold", "", "", ""],
        #         ["", "Reconstructor", "0.999", "1", "", "", ""],
        #         ["Bundle adjustment", "Bundle adjustment", "x_threshold", "y_threshold", "", "", ""],
        #         ["", "Bundle adjustment", "0.5", "1", "", "", ""]]
        # self.algorithm_parameters_table.setColumnCount(len(text[0]))
        # self.algorithm_parameters_table.setRowCount(len(text))
        # self.algorithm_parameters_table.setHorizontalHeaderLabels(['Step', 'Algorithm', 'param_1', 'param_2',
        #                                                            'param_3',
        #                                                            'param_4', 'param_5'])
        #
        # header = self.algorithm_parameters_table.horizontalHeader()
        # header.setSectionResizeMode(0, QHeaderView.Stretch)
        # for i in range(1, 7):
        #     header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        #
        # for i in range(len(text)):
        #     for j in range(len(text[i])):
        #         item = QTableWidgetItem(text[i][j])
        #         self.algorithm_parameters_table.setItem(i, j, item)

        # test_text = ["text_1", "text_2", "text_3"]
        # for i in range(len(test_text)):
        #     # item.setFlags(item.flags() | Qt.ItemIsSelectable)
        #

    def __choose_files_and_fill_image_table(self):
        image_files = self.__open_file_dialog(title=window_const.IMAGES_FOR_PROCESS_DIALOG_TITLE,
                                              directory=window_const.IMAGES_FOR_PROCESS_DIALOG_DIR,
                                              file_filter=window_const.IMAGES_FOR_PROCESS_DIALOG_FILE_FILTER)
        if image_files:
            self.path_to_image_files = os.path.dirname(image_files[0])
            image_names = [os.path.basename(path) for path in image_files]
            self.__fill_image_table(image_names)

    def __open_file_dialog(self, title="", directory="", file_filter=None):
        """Open file dialog and return file paths of the selected files.

        Args:
            title (str): title of the file dialog
            directory (str): path to start dir for opening
            file_filter (str): set of file filters
        Returns:
            (list[str]): list of the paths to selected files.
        """

        open_dialog = QFileDialog()
        file_paths = open_dialog.getOpenFileNames(self, title, directory, file_filter)
        # file_paths is tuple like (file_paths, format)
        return file_paths[0]

    def __fill_image_table(self, image_names):
        """Fill image table using image names."""

        self.image_table.clearContents()
        num_of_images = len(image_names)
        self.image_table.setRowCount(num_of_images)
        for i in range(num_of_images):
            # insert order number
            order = str(i+1)
            order_item = QTableWidgetItem(order)
            self.image_table.setItem(i, 0, order_item)

            # insert image name
            image_name_item = QTableWidgetItem(image_names[i])
            image_name_item.setFlags(Qt.ItemIsEnabled)
            self.image_table.setItem(i, 1, image_name_item)

            # insert remove button
            remove_button_sell = QPushButton('X')
            remove_button_sell.clicked.connect(self.__remove_item_from_image_table)
            self.image_table.setCellWidget(i, 2, remove_button_sell)

    # todo: add ограничения на ввод только чисел в номере

    def __remove_item_from_image_table(self):
        """Remove row of clicked button in image table"""

        button = qApp.focusWidget()
        index = self.image_table.indexAt(button.pos())
        if index.isValid():
            order_of_removing_image = index.row()
            self.image_table.removeRow(order_of_removing_image)
            # change other order numbers
            for i in range(self.image_table.rowCount()):
                # get order number
                item_text = self.image_table.item(i, 0).text()
                item_order_number = int(item_text)
                if item_order_number > order_of_removing_image:
                    item_order_number -= 1
                    self.image_table.item(i, 0).setText(str(item_order_number))



    def get_next_image_number(self):
        pass

    def load_image(self):
        pass

    def load_images(self):
        pass

    def change_order_in_image_table(self):
        pass

    def init_combobox(self):
        pass

    def set_default_algorithm(self):
        pass

    # error window
    # message window
    # show_model

    # todo: Main
    # todo: add parameters setting
    # todo: add algorithm choosing by combobox
    # todo: may be buttons add "Add images" and "Clear" instead of "Load images"
    # todo: add camera params input (f, sw, sh, get_camera_params)

    # todo: Process button
    # todo: process all images from list
    # todo: SORT TABLE AFTER CHANGE ORDER
    # Todo: все числа между старым и новым уменьшатся на 1
