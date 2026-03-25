import numpy as np
from typing import Dict, Any
import math

# TODO: Rewrite This Class
class CameraCalibration:
    def __init__(self, calibration_data: Dict[str, Any]):
        self.focal_length = calibration_data['focalLength']  # мм
        self.mount_height = calibration_data['mountHeight']  # м
        self.tilt_angle = math.radians(calibration_data['tiltAngle'])  # рад
        self.sensor_width = calibration_data['sensorWidth']  # мм
        self.image_width = calibration_data['imageWidth']  # пикс

        # Предварительный расчет коэффициентов
        self._calculate_scaling_factors()

    def _calculate_scaling_factors(self):
        """Расчет коэффициентов масштабирования"""
        # Горизонтальный угол обзора (в радианах)
        self.horizontal_fov = 2 * math.atan2(self.sensor_width / 2, self.focal_length)

        # Коэффициент для конвертации пикселей в метры по горизонтали
        # При угле наклона 90 градусов (сверху вниз)
        if self.tilt_angle == math.pi / 2:  # 90 градусов
            # Прямая проекция сверху
            ground_width_at_distance = 2 * self.mount_height * math.tan(self.horizontal_fov / 2)
            self.pixels_to_meters_horizontal = ground_width_at_distance / self.image_width
        else:
            # При наклонной съемке - более сложный расчет
            # Упрощенная модель: учитываем перспективу
            self.pixels_to_meters_horizontal = (self.sensor_width / self.focal_length) * \
                                               self.mount_height / self.image_width

    def pixels_to_square_meters(self, pixel_count: int) -> float:
        pixels_per_meter = 1 / self.pixels_to_meters_horizontal
        area_in_pixels = pixel_count

        area_sqm = area_in_pixels * (self.pixels_to_meters_horizontal ** 2)

        return round(area_sqm, 2)

    def get_scaling_factor(self) -> float:
        return self.pixels_to_meters_horizontal


class CalibrationError(Exception):
    # десь можно добавить ошиьку
    pass