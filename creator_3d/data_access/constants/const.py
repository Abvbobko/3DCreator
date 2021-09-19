import os

SENSOR_DATABASE_FOLDER = './data_access'
SENSOR_DATABASE_FILE_NAME = 'sensor_database.csv'
SENSOR_DATABASE_PATH = os.path.join(SENSOR_DATABASE_FOLDER, SENSOR_DATABASE_FILE_NAME)


class CsvHeaderParamName:
    camera_model = 'CameraModel'
    sensor_width = 'SensorWidth(mm)'
    sensor_height = 'SensorHeight(mm)'


OBJ_EXTENSION = "obj"
PLY_EXTENSION = "ply"
