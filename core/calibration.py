import numpy as np
from typing import Dict, Any
import math
import logging

logger = logging.getLogger(__name__)


class CameraCalibration:
    def __init__(self, calibration_data: Dict[str, Any]):
        self.focal_length = calibration_data['focalLength']  # мм
        self.mount_height = calibration_data['mountHeight']  # м
        self.tilt_angle_deg = calibration_data['tiltAngle']  # градусы
        self.tilt_angle_rad = math.radians(self.tilt_angle_deg)  # радианы
        self.sensor_width = calibration_data['sensorWidth']  # мм
        self.image_width = calibration_data['imageWidth']  # пикс

        # Дополнительные параметры
        self.sensor_height = self.sensor_width * (9 / 16)  # Предполагаем соотношение 16:9
        self.image_height = int(self.image_width * (9 / 16))  # Высота изображения для 16:9

        # Предварительный расчет
        self._calculate_camera_parameters()

    def _calculate_camera_parameters(self):
        """Расчет параметров камеры для перспективной проекции"""

        # Вертикальный угол обзора (FOV)
        self.vertical_fov = 2 * math.atan2(self.sensor_height / 2, self.focal_length)

        # Горизонтальный угол обзора
        self.horizontal_fov = 2 * math.atan2(self.sensor_width / 2, self.focal_length)

        # Расстояние до центра изображения по земле (при наклонной съемке)
        # Формула: distance = mount_height / tan(tilt_angle)
        if self.tilt_angle_rad > 0 and self.tilt_angle_rad < math.pi:
            self.center_distance = self.mount_height / math.tan(self.tilt_angle_rad)
        else:
            # При вертикальной съемке (90 градусов)
            self.center_distance = 0

        logger.debug(f"Camera params - HFOV: {math.degrees(self.horizontal_fov):.2f}°, "
                     f"VFOV: {math.degrees(self.vertical_fov):.2f}°, "
                     f"Center distance: {self.center_distance:.2f}m")

    def pixels_to_square_meters(self, pixel_count: int) -> float:
        """
        Конвертация количества пикселей в площадь в квадратных метрах
        с учетом перспективных искажений
        """
        # Если все пиксели - это вся площадь кадра
        if pixel_count == self.image_width * self.image_height:
            return self._calculate_total_frame_area()

        # Для частичной площади используем пропорцию с весами
        # Создаем карту весов для каждого пикселя
        total_weighted_pixels = self._calculate_total_weighted_pixels()

        # Вычисляем взвешенную площадь
        if total_weighted_pixels > 0:
            total_frame_area = self._calculate_total_frame_area()
            weighted_area = (pixel_count / total_weighted_pixels) * total_frame_area
            return round(weighted_area, 2)
        else:
            return 0.0

    def _calculate_total_frame_area(self) -> float:
        """Вычисление общей площади кадра в квадратных метрах"""
        try:
            # Для наклонной камеры используем интегральный подход
            if self.tilt_angle_deg < 90 and self.tilt_angle_deg > 0:
                return self._calculate_tilted_frame_area()
            else:
                # Для вертикальной съемки (сверху вниз)
                return self._calculate_vertical_frame_area()
        except Exception as e:
            logger.error(f"Error calculating frame area: {e}")
            # Fallback: упрощенный расчет
            return self._calculate_simple_frame_area()

    def _calculate_tilted_frame_area(self) -> float:
        """
        Расчет площади кадра для наклонной камеры
        Использует интегрирование по углу обзора
        """
        # Углы обзора от верхнего края до нижнего
        # Верхний угол (дальняя граница)
        top_angle_rad = self.tilt_angle_rad - (self.vertical_fov / 2)
        # Нижний угол (ближняя граница)
        bottom_angle_rad = self.tilt_angle_rad + (self.vertical_fov / 2)

        # Расстояния до верхней и нижней границ
        if top_angle_rad > 0 and top_angle_rad < math.pi:
            top_distance = self.mount_height / math.tan(top_angle_rad)
        else:
            top_distance = 1000  # Очень большое расстояние для горизонтальной линии

        if bottom_angle_rad > 0 and bottom_angle_rad < math.pi:
            bottom_distance = self.mount_height / math.tan(bottom_angle_rad)
        else:
            bottom_distance = self.mount_height / 0.001  # Очень маленький угол

        # Ширина кадра на разных расстояниях
        top_width = 2 * top_distance * math.tan(self.horizontal_fov / 2)
        bottom_width = 2 * bottom_distance * math.tan(self.horizontal_fov / 2)

        # Высота кадра на земле (разница расстояний)
        frame_height = abs(bottom_distance - top_distance)

        # Площадь как трапеция
        if frame_height > 0:
            area = ((top_width + bottom_width) / 2) * frame_height
            logger.debug(f"Tilted frame area calculation - "
                         f"Top distance: {top_distance:.2f}m, "
                         f"Bottom distance: {bottom_distance:.2f}m, "
                         f"Top width: {top_width:.2f}m, "
                         f"Bottom width: {bottom_width:.2f}m, "
                         f"Frame height: {frame_height:.2f}m, "
                         f"Area: {area:.2f} sqm")
            return area
        else:
            return self._calculate_simple_frame_area()

    def _calculate_vertical_frame_area(self) -> float:
        """Расчет площади для вертикальной съемки (сверху вниз)"""
        # Ширина кадра на земле
        frame_width = 2 * self.mount_height * math.tan(self.horizontal_fov / 2)

        # Высота кадра на земле
        frame_height = 2 * self.mount_height * math.tan(self.vertical_fov / 2)

        area = frame_width * frame_height
        logger.debug(f"Vertical frame area: {area:.2f} sqm (width: {frame_width:.2f}m, height: {frame_height:.2f}m)")
        return area

    def _calculate_simple_frame_area(self) -> float:
        """Упрощенный расчет площади для fallback"""
        # Предполагаем, что кадр покрывает определенную площадь
        # Используем среднее расстояние до центра
        if self.tilt_angle_rad > 0:
            distance = self.mount_height / math.sin(self.tilt_angle_rad)
        else:
            distance = self.mount_height

        frame_width = 2 * distance * math.tan(self.horizontal_fov / 2)
        frame_height = 2 * distance * math.tan(self.vertical_fov / 2)

        return frame_width * frame_height

    def _calculate_total_weighted_pixels(self) -> float:
        """
        Вычисление суммы весов всех пикселей для взвешенного расчета площади
        """
        # Создаем упрощенную модель весов
        # Пиксели ближе к камере (нижняя часть кадра) имеют больший вес
        # Пиксели дальше (верхняя часть) имеют меньший вес

        total_weight = 0
        for y in range(self.image_height):
            # Нормализованная вертикальная координата (0 = верх, 1 = низ)
            v = y / self.image_height

            # Угол для этой строки
            angle_rad = self.tilt_angle_rad + (self.vertical_fov * (v - 0.5))

            # Вес пропорционален площади на земле
            if angle_rad > 0 and angle_rad < math.pi:
                distance = self.mount_height / math.tan(angle_rad)
                # Вес строки пропорционален ширине на этом расстоянии
                row_weight = distance * math.tan(self.horizontal_fov / 2)
            else:
                row_weight = 1

            total_weight += row_weight * self.image_width

        return total_weight

    def get_scaling_factor(self) -> float:
        """Возвращает средний коэффициент масштабирования"""
        total_area = self._calculate_total_frame_area()
        total_pixels = self.image_width * self.image_height
        return total_area / total_pixels if total_pixels > 0 else 0