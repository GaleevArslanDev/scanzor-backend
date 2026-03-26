import numpy as np
from typing import Dict, Any, Optional
from .calibration import CameraCalibration
import logging
import base64
import cv2

logger = logging.getLogger(__name__)


class AreaCalculator:
    def __init__(self, calibration: CameraCalibration):
        self.calibration = calibration
        # Предварительно вычисляем карту весов для каждого пикселя
        self.weight_map = self._calculate_weight_map()

    def _calculate_weight_map(self) -> np.ndarray:
        """
        Создает карту весов для каждого пикселя изображения.
        Вес пикселя = реальная площадь на земле, которую он представляет.
        """
        height = self.calibration.image_height
        width = self.calibration.image_width

        # Создаем карту весов
        weight_map = np.zeros((height, width), dtype=np.float64)

        # Для каждой строки (y-координата) вычисляем площадь, которую представляет эта строка
        for y in range(height):
            # Нормализованная вертикальная координата (0 = верх, 1 = низ)
            v = y / height

            # Угол от горизонта до этой строки
            # Используем линейную интерполяцию угла по вертикали
            angle_rad = self.calibration._get_angle_for_row(v)

            # Расстояние до этой строки на земле
            if angle_rad > 0.01 and angle_rad < np.pi - 0.01:
                distance = self.calibration.mount_height / np.tan(angle_rad)
            else:
                distance = 1000  # Для очень далеких объектов

            # Ширина на этом расстоянии
            row_width_m = 2 * distance * np.tan(self.calibration.horizontal_fov / 2)

            # Высота, которую представляет этот ряд пикселей
            # Используем разницу расстояний между соседними строками
            v_next = min(1.0, v + 1.0 / height)
            angle_next = self.calibration._get_angle_for_row(v_next)

            if angle_next > 0.01 and angle_next < np.pi - 0.01:
                distance_next = self.calibration.mount_height / np.tan(angle_next)
            else:
                distance_next = distance

            row_height_m = abs(distance_next - distance)

            # Площадь одного пикселя в этой строке
            pixel_area_m2 = (row_width_m / width) * row_height_m

            # Заполняем всю строку одинаковым весом
            weight_map[y, :] = pixel_area_m2

        # Нормализация для проверки
        total_calculated_area = np.sum(weight_map)
        actual_frame_area = self.calibration._calculate_total_frame_area()

        logger.info(f"Total calculated area via weight map: {total_calculated_area:.2f} sqm")
        logger.info(f"Actual frame area: {actual_frame_area:.2f} sqm")
        logger.info(f"Ratio: {total_calculated_area / actual_frame_area:.4f}")

        # Небольшая корректировка для компенсации погрешностей
        if abs(total_calculated_area - actual_frame_area) > 0.1:
            scale_factor = actual_frame_area / total_calculated_area
            weight_map *= scale_factor
            logger.info(f"Applied scale correction: {scale_factor:.4f}")

        return weight_map

    def calculate_area(self, mask: np.ndarray, return_mask: bool = True) -> Dict[str, Any]:
        """
        Расчет площади с использованием карты весов
        """
        total_pixels = mask.size
        processed_pixels = int(np.sum(mask))

        # Процент обработанной площади (по количеству пикселей)
        processed_percentage = (processed_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        # Расчет реальной площади с использованием весов
        weighted_area = np.sum(self.weight_map * mask)
        total_area_sqm = round(weighted_area, 2)

        # Альтернативный расчет для верификации
        total_frame_area = self.calibration._calculate_total_frame_area()
        total_weighted_pixels = np.sum(self.weight_map)

        logger.info(f"Processed pixels: {processed_pixels}/{total_pixels} "
                    f"({processed_percentage:.2f}%)")
        logger.info(f"Processed weighted area: {total_area_sqm:.2f} sqm")
        logger.info(f"Total frame area: {total_frame_area:.2f} sqm")
        logger.info(f"Total weighted pixels sum: {total_weighted_pixels:.2f}")

        # Статистика по весам для отладки
        weights_in_mask = self.weight_map[mask > 0]
        if len(weights_in_mask) > 0:
            logger.info(f"Weight stats in mask - min: {np.min(weights_in_mask):.6f}, "
                        f"max: {np.max(weights_in_mask):.6f}, "
                        f"mean: {np.mean(weights_in_mask):.6f}")

        result = {
            'total_area_sqm': total_area_sqm,
            'processed_percentage': round(processed_percentage, 2),
            'processed_pixels': processed_pixels,
            'total_pixels': total_pixels
        }

        # Добавляем отладочную информацию
        result['debug_info'] = {
            'scaling_factor': self.calibration.get_scaling_factor(),
            'total_frame_area': total_frame_area,
            'total_weighted_pixels': round(total_weighted_pixels, 2),
            'weighted_area_total': round(np.sum(self.weight_map), 2),
            'tilt_angle': self.calibration.tilt_angle_deg,
            'mount_height': self.calibration.mount_height,
            'vertical_fov_deg': round(np.degrees(self.calibration.vertical_fov), 2),
            'horizontal_fov_deg': round(np.degrees(self.calibration.horizontal_fov), 2),
            'min_pixel_weight': round(np.min(self.weight_map), 6),
            'max_pixel_weight': round(np.max(self.weight_map), 6),
            'weight_ratio': round(np.max(self.weight_map) / np.min(self.weight_map), 2)
        }

        if return_mask:
            result['mask'] = self._encode_mask_to_base64(mask)

        return result

    def _encode_mask_to_base64(self, mask: np.ndarray) -> str:
        """Кодирует бинарную маску в base64 строку"""
        # Преобразуем маску в изображение (0-255)
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Кодируем в PNG
        _, buffer = cv2.imencode('.png', mask_uint8)

        # Конвертируем в base64
        mask_base64 = base64.b64encode(buffer).decode('utf-8')

        return f"data:image/png;base64,{mask_base64}"