import numpy as np
from typing import Dict, Any, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class CameraCalibration:
    def __init__(self, calibration_data: Dict[str, Any], image_shape: Tuple[int, int] = None):
        """
        Args:
            calibration_data: Данные калибровки
            image_shape: (height, width) реальные размеры изображения
        """
        self.focal_length = calibration_data['focalLength']  # мм
        self.mount_height = calibration_data['mountHeight']  # м
        self.tilt_angle_deg = calibration_data['tiltAngle']  # градусы
        self.tilt_angle_rad = math.radians(self.tilt_angle_deg)  # радианы
        self.sensor_width = calibration_data['sensorWidth']  # мм
        self.image_width = calibration_data['imageWidth']  # пикс (ширина из калибровки)

        # Используем реальные размеры изображения, если они переданы
        if image_shape is not None:
            self.image_height = image_shape[0]
            self.real_image_width = image_shape[1]
            logger.info(f"Using actual image dimensions: {self.real_image_width}x{self.image_height}")
        else:
            # Fallback: вычисляем высоту на основе соотношения сторон 4:3
            self.aspect_ratio = 4 / 3
            self.image_height = int(self.image_width / self.aspect_ratio)
            self.real_image_width = self.image_width
            logger.warning(f"Using calculated image dimensions: {self.real_image_width}x{self.image_height}")

        # Расчет сенсора на основе реального соотношения сторон изображения
        self.sensor_height = self.sensor_width * (self.image_height / self.real_image_width)

        logger.info(f"Image dimensions: {self.real_image_width}x{self.image_height}")
        logger.info(f"Sensor dimensions: {self.sensor_width:.2f}x{self.sensor_height:.2f} mm")

        # Предварительный расчет
        self._calculate_camera_parameters()

    def _calculate_camera_parameters(self):
        """Расчет параметров камеры для перспективной проекции"""
        # Вертикальный угол обзора (FOV)
        self.vertical_fov = 2 * math.atan2(self.sensor_height / 2, self.focal_length)

        # Горизонтальный угол обзора
        self.horizontal_fov = 2 * math.atan2(self.sensor_width / 2, self.focal_length)

        # Расстояние до центра изображения по земле (при наклонной съемке)
        if self.tilt_angle_rad > 0.01 and self.tilt_angle_rad < math.pi - 0.01:
            self.center_distance = self.mount_height / math.tan(self.tilt_angle_rad)
        else:
            self.center_distance = 0

        logger.info(f"Camera params - HFOV: {math.degrees(self.horizontal_fov):.2f}°, "
                    f"VFOV: {math.degrees(self.vertical_fov):.2f}°, "
                    f"Center distance: {self.center_distance:.2f}m")

    def _get_angle_for_row(self, v: float) -> float:
        """
        Вычисляет угол от горизонта до строки с нормализованной координатой v
        v = 0 - верх изображения (дальняя граница)
        v = 1 - низ изображения (ближняя граница)
        """
        # Угол для центра изображения
        center_angle = self.tilt_angle_rad

        # Половина вертикального поля зрения
        half_vfov = self.vertical_fov / 2

        # Линейная интерполяция угла
        # v = 0 -> angle = center_angle - half_vfov (верхний край)
        # v = 1 -> angle = center_angle + half_vfov (нижний край)
        angle = center_angle - half_vfov + (v * self.vertical_fov)

        # Ограничиваем угол
        angle = max(0.01, min(angle, math.pi - 0.01))

        return angle

    def _calculate_total_frame_area(self) -> float:
        """Вычисление общей площади кадра в квадратных метрах"""
        try:
            # Интегрируем по всей площади изображения
            total_area = 0.0

            # Используем точное количество строк для интегрирования
            num_samples = self.image_height

            for y in range(num_samples):
                v = y / num_samples
                angle = self._get_angle_for_row(v)

                # Расстояние до этой строки
                if angle > 0.01 and angle < math.pi - 0.01:
                    distance = self.mount_height / math.tan(angle)
                else:
                    distance = 1000

                # Ширина на этом расстоянии
                row_width = 2 * distance * math.tan(self.horizontal_fov / 2)

                # Высота полосы на земле (дифференциал)
                v_next = min(1.0, v + 1.0 / num_samples)
                angle_next = self._get_angle_for_row(v_next)

                if angle_next > 0.01 and angle_next < math.pi - 0.01:
                    distance_next = self.mount_height / math.tan(angle_next)
                else:
                    distance_next = distance

                row_height = abs(distance_next - distance)

                # Площадь полосы
                strip_area = row_width * row_height
                total_area += strip_area

            logger.info(f"Calculated total frame area: {total_area:.2f} sqm")
            return total_area

        except Exception as e:
            logger.error(f"Error calculating frame area: {e}")
            return self._calculate_simple_frame_area()

    def _calculate_simple_frame_area(self) -> float:
        """Упрощенный расчет площади для fallback"""
        # Используем среднее расстояние
        if self.tilt_angle_rad > 0.01 and self.tilt_angle_rad < math.pi - 0.01:
            distance = self.mount_height / math.sin(self.tilt_angle_rad)
        else:
            distance = self.mount_height

        frame_width = 2 * distance * math.tan(self.horizontal_fov / 2)
        frame_height = 2 * distance * math.tan(self.vertical_fov / 2)

        area = frame_width * frame_height
        logger.warning(f"Using simplified area calculation: {area:.2f} sqm")
        return area

    def get_scaling_factor(self) -> float:
        """Возвращает средний коэффициент масштабирования"""
        total_area = self._calculate_total_frame_area()
        total_pixels = self.image_height * self.real_image_width
        return total_area / total_pixels if total_pixels > 0 else 0


class CalibrationError(Exception):
    pass