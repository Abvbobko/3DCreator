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
from PyQt5.QtCore import (Qt, QRegExp, QDate)
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
        self.__set_combobox(combobox=self.feature_extraction_combobox,
                            step_name=self.main_controller.get_extract_step_name(),
                            default_algorithm=self.main_controller.get_extract_step_default_algorithm())

        self.__set_combobox(combobox=self.feature_matching_combobox,
                            step_name=self.main_controller.get_match_step_name(),
                            default_algorithm=self.main_controller.get_match_step_default_algorithm())

        self.__set_combobox(combobox=self.reconstruction_combobox,
                            step_name=self.main_controller.get_reconstruct_step_name(),
                            default_algorithm=self.main_controller.get_reconstruct_default_algorithm())

        self.__set_combobox(combobox=self.bundle_adjustment_combobox,
                            step_name=self.main_controller.get_bundle_adjust_step_name(),
                            default_algorithm=self.main_controller.get_bundle_adjust_default_algorithm())

        self.feature_extraction_combobox.currentTextChanged.connect(self.__change_extraction_step_algorithm)
        self.feature_matching_combobox.currentTextChanged.connect(self.__change_matching_step_algorithm)
        self.reconstruction_combobox.currentTextChanged.connect(self.__change_reconstruction_step_algorithm)
        self.bundle_adjustment_combobox.currentTextChanged.connect(self.__change_bundle_adjustment_step_algorithm)

        # camera parameters edits
        self.__set_string_edit(edit=self.focal_length_edit,
                               field_name=window_const.FOCAL_LENGTH_EDIT_CONST.field_name,
                               max_length=window_const.FOCAL_LENGTH_EDIT_CONST.max_length,
                               mask_regex=window_const.FOCAL_LENGTH_EDIT_CONST.mask_regex,
                               can_be_empty=window_const.FOCAL_LENGTH_EDIT_CONST.can_be_empty)

        self.__set_string_edit(edit=self.sensor_width_edit,
                               field_name=window_const.SENSOR_WIDTH_EDIT_CONST.field_name,
                               max_length=window_const.SENSOR_WIDTH_EDIT_CONST.max_length,
                               mask_regex=window_const.SENSOR_WIDTH_EDIT_CONST.mask_regex,
                               can_be_empty=window_const.SENSOR_WIDTH_EDIT_CONST.can_be_empty)

        self.__set_string_edit(edit=self.sensor_height_edit,
                               field_name=window_const.SENSOR_HEIGHT_EDIT_CONST.field_name,
                               max_length=window_const.SENSOR_HEIGHT_EDIT_CONST.max_length,
                               mask_regex=window_const.SENSOR_HEIGHT_EDIT_CONST.mask_regex,
                               can_be_empty=window_const.SENSOR_HEIGHT_EDIT_CONST.can_be_empty)

        self.__set_string_edit(edit=self.image_width_edit,
                               field_name=window_const.IMAGE_WIDTH_EDIT_CONST.field_name,
                               max_length=window_const.IMAGE_WIDTH_EDIT_CONST.max_length,
                               mask_regex=window_const.IMAGE_WIDTH_EDIT_CONST.mask_regex,
                               can_be_empty=window_const.IMAGE_WIDTH_EDIT_CONST.can_be_empty)

        self.__set_string_edit(edit=self.image_height_edit,
                               field_name=window_const.IMAGE_HEIGHT_EDIT_CONST.field_name,
                               max_length=window_const.IMAGE_HEIGHT_EDIT_CONST.max_length,
                               mask_regex=window_const.IMAGE_HEIGHT_EDIT_CONST.mask_regex,
                               can_be_empty=window_const.IMAGE_HEIGHT_EDIT_CONST.can_be_empty)

        # exif loading
        self.load_params_from_exif_button.clicked.connect(self.__choose_file_and_load_camera_params)

        self.set_default_algorithms_button.clicked.connect(self.__set_default_algorithms)

        # algorithm params setting
        self.__fill_algorithm_table()

    @staticmethod
    def __get_item_max_length(list_of_items):
        if not list_of_items:
            raise Exception("list_of_items must have at least 1 item.")
        max_len = len(list_of_items[0])
        max_len_index = 0
        for i in range(1, len(list_of_items)):
            if len(list_of_items[i]) >= max_len:
                max_len = len(list_of_items[i])
                max_len_index = i
        return len(list_of_items[max_len_index])

    @staticmethod
    def __create_table_item(text, is_editable=True):
        item = QTableWidgetItem(str(text))
        if not is_editable:
            item.setFlags(Qt.ItemIsEnabled)
        return item

    @staticmethod
    def __create_empty_table_cell(is_editable=False):
        return MainWindow.__create_table_item('', is_editable)

    def __change_combobox_step_algorithm(self, combobox, new_algorithm):
        self.__refill_step_in_algorithm_table(combobox.step_name, new_algorithm)

    def __change_extraction_step_algorithm(self, value):        
        self.__change_combobox_step_algorithm(self.feature_extraction_combobox, value)

    def __change_matching_step_algorithm(self, value):
        self.__change_combobox_step_algorithm(self.feature_matching_combobox, value)

    def __change_reconstruction_step_algorithm(self, value):
        self.__change_combobox_step_algorithm(self.reconstruction_combobox, value)

    def __change_bundle_adjustment_step_algorithm(self, value):
        self.__change_combobox_step_algorithm(self.bundle_adjustment_combobox, value)

    def __get_step_comboboxes_list(self):
        return [self.feature_extraction_combobox,
                self.feature_matching_combobox,
                self.reconstruction_combobox,
                self.bundle_adjustment_combobox]

    @staticmethod
    def __change_table_cols_number(table, new_count_of_cols):
        num_of_cols = table.columnCount()
        if new_count_of_cols < num_of_cols:
            delta = num_of_cols - new_count_of_cols
            for _ in range(delta):
                last_column_index = table.columnCount() - 1
                table.removeColumn(last_column_index)
        elif new_count_of_cols > num_of_cols:
            delta = new_count_of_cols - num_of_cols
            table.setColumnCount(new_count_of_cols)
            for j_delta in range(delta):
                for i in range(table.rowCount()):
                    table.setItem(i, num_of_cols + j_delta,
                                  MainWindow.__create_empty_table_cell())

    @staticmethod
    def __narrow_table_if_needed(table):
        is_last_column_empty = True
        while is_last_column_empty:
            last_column_index = table.columnCount() - 1
            count_of_empty_rows = 0
            row_count = table.rowCount()
            for i in range(row_count):
                cell_text = table.item(i, last_column_index).text()
                if not cell_text:
                    count_of_empty_rows += 1
            if count_of_empty_rows == row_count:
                table.removeColumn(last_column_index)
            else:
                is_last_column_empty = False

    @staticmethod
    def __get_step_line(table, step_name):
        for i in range(0, table.rowCount(), 2):
            print(table.item(i, 0).text())
            if table.item(i, 0).text() == step_name:
                return i
        return None

    def __change_step_default_params(self, step_name, algorithm_name, step_params):
        step_row = self.__get_step_line(self.algorithm_parameters_table, step_name)
        num_of_cols = self.algorithm_parameters_table.columnCount()
        algorithm_params = list(step_params)
        print("step_row", step_row)
        self.algorithm_parameters_table.setItem(step_row, 1,
                                                self.__create_table_item(algorithm_name, False))

        for j in range(2, num_of_cols):
            if j - 2 >= len(step_params):
                item_1 = self.__create_empty_table_cell()
                item_2 = self.__create_empty_table_cell()
            else:
                item_1 = self.__create_table_item(algorithm_params[j - 2], False)
                item_2 = self.__create_table_item(step_params[algorithm_params[j - 2]])
            self.algorithm_parameters_table.setItem(step_row, j, item_1)
            self.algorithm_parameters_table.setItem(step_row + 1, j, item_2)

    def __refill_step_in_algorithm_table(self, step_name, algorithm_name):
        step_default_params = self.main_controller.get_step_algorithm_default_params(step_name, algorithm_name)
        # if number of params is higher than num of cols
        if len(step_default_params) + 2 > self.algorithm_parameters_table.columnCount():
            new_num_of_cols = len(step_default_params) + 2
            self.__change_table_cols_number(self.algorithm_parameters_table, new_num_of_cols)
        self.__change_step_default_params(step_name, algorithm_name, step_default_params)
        self.__narrow_table_if_needed(self.algorithm_parameters_table)

    def __fill_algorithm_table(self):
        step_comboboxes = self.__get_step_comboboxes_list()
        all_steps_params = []
        for step_combobox in step_comboboxes:
            all_steps_params.append(
                self.main_controller.get_step_algorithm_default_params(step_combobox.step_name,
                                                                       str(step_combobox.currentText()))
            )
        # set number of col, rows and header of the table
        num_of_cols = self.__get_item_max_length(all_steps_params)
        header = ['Step', 'Algorithm'] + [f"param_{i}" for i in range(num_of_cols)]
        num_of_cols = len(header)
        self.algorithm_parameters_table.setColumnCount(num_of_cols)
        self.algorithm_parameters_table.setRowCount(2 * len(step_comboboxes))
        self.algorithm_parameters_table.setHorizontalHeaderLabels(header)
        table_header_obj = self.algorithm_parameters_table.horizontalHeader()
        for i in range(len(header)):
            table_header_obj.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        # fill table
        for i in range(len(step_comboboxes)):
            # set step name
            self.algorithm_parameters_table.setItem(i * 2, 0,
                                                    self.__create_table_item(step_comboboxes[i].step_name, False))

            # empty cell before step cell
            self.algorithm_parameters_table.setItem(i * 2 + 1, 0,
                                                    self.__create_empty_table_cell())

            # set algorithm name
            self.algorithm_parameters_table.setItem(i * 2, 1,
                                                    self.__create_table_item(step_comboboxes[i].currentText(), False))

            # empty cell before algorithm name cell
            self.algorithm_parameters_table.setItem(i * 2 + 1, 1,
                                                    self.__create_empty_table_cell())

            # set params
            algorithm_params = list(all_steps_params[i])
            for j in range(2, num_of_cols):
                if j-2 >= len(algorithm_params):
                    item_1 = self.__create_empty_table_cell()
                    item_2 = self.__create_empty_table_cell()
                else:
                    item_1 = self.__create_table_item(algorithm_params[j-2], False)
                    item_2 = self.__create_table_item(all_steps_params[i][algorithm_params[j-2]])
                self.algorithm_parameters_table.setItem(i * 2, j, item_1)
                self.algorithm_parameters_table.setItem(i * 2 + 1, j, item_2)

    def __set_default_algorithms(self):
        comboboxes = self.__get_step_comboboxes_list()

        for combobox in comboboxes:
            default_algorithm = combobox.default_algorithm
            if default_algorithm:
                index = combobox.findText(default_algorithm)
                if index >= 0:
                    combobox.setCurrentIndex(index)
        # todo: add default params of algorithm setting

    def __set_combobox(self, combobox, step_name, default_algorithm=''):
        combobox.step_name = step_name
        combobox.default_algorithm = default_algorithm
        combobox.clear()
        combobox.addItems(self.main_controller.get_algorithm_names_for_step(combobox.step_name))
        if default_algorithm:
            index = combobox.findText(default_algorithm)
            if index >= 0:
                combobox.setCurrentIndex(index)

    @staticmethod
    def __set_string_edit(edit,
                          field_name=None,
                          can_be_empty=False,
                          max_length=255,
                          mask_regex=None,
                          placeholder=None):
        """Set parameters to the edit"""

        edit.mask_regex = mask_regex
        edit.setMaxLength(max_length)
        if mask_regex:
            edit.setValidator(QtGui.QRegExpValidator(QRegExp(edit.mask_regex)))
        edit.field_name = field_name
        edit.can_be_empty = can_be_empty
        if placeholder:
            edit.setPlaceholderText(placeholder)

    def __choose_file_and_load_camera_params(self):
        image_files = self.__open_file_dialog(title=window_const.EXIF_IMAGE_DIALOG_TITLE,
                                              directory=window_const.EXIF_IMAGE_DIALOG_DIR,
                                              file_filter=window_const.EXIF_IMAGE_DIALOG_FILE_FILTER)
        if image_files:
            image_file_path = image_files[0]
            camera = self.__load_params_from_exif_image(image_file_path)
            self.__fill_camera_params_edits(camera)
            self.__show_camera_model_and_empty_params(camera)

    @staticmethod
    def call_message_box(title="", text="", icon=QMessageBox.NoIcon):
        message_box = QMessageBox()
        message_box.setIcon(icon)
        message_box.setText(text)
        message_box.setWindowTitle(title)
        message_box.exec_()

    @staticmethod
    def call_error_box(error_title="Error", error_text=""):
        print("ERROR")
        MainWindow.call_message_box(error_title, error_text, QMessageBox.Critical)

    @staticmethod
    def call_ok_box(ok_title="ОK", ok_text=""):
        print("OK")
        MainWindow.call_message_box(ok_title, ok_text, QMessageBox.Information)

    def __show_camera_model_and_empty_params(self, camera):
        """Call message box with camera model and unrecognized params"""

        if camera.model:
            camera_model_text = camera.model
        else:
            camera_model_text = "not recognized"

        params = {
            camera.focal_length_param_name: camera.focal_length,
            camera.sensor_size_param_name[0]: camera.sensor_size[0],
            camera.sensor_size_param_name[1]: camera.sensor_size[1],
            camera.image_size_param_name[0]: camera.image_size[0],
            camera.image_size_param_name[1]: camera.image_size[0]
        }

        empty_params = []
        for param_name in params:
            if not params[param_name]:
                empty_params.append(param_name)

        if empty_params:
            empty_params_text = "\nNot recognized params:\n%s." % ",\n".join(empty_params)
        else:
            empty_params_text = ""

        message_text = f"Camera model: {camera_model_text}.{empty_params_text}"
        self.call_ok_box(ok_text=message_text)

    def __load_params_from_exif_image(self, image_path):
        return self.main_controller.get_params_from_exif_image(image_path)

    def __fill_camera_params_edits(self, camera):
        # test: test when not all camera params are available
        self.focal_length_edit.setText(str(camera.focal_length))
        sensor_size = camera.sensor_size
        self.sensor_width_edit.setText(str(sensor_size[0]))
        self.sensor_height_edit.setText(str(sensor_size[1]))
        image_size = camera.image_size
        self.image_width_edit.setText(str(image_size[0]))
        self.image_height_edit.setText(str(image_size[1]))

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
            order = str(i + 1)
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

    # error window
    # message window
    # show_model

    # Todo: все числа между старым и новым уменьшатся на 1
