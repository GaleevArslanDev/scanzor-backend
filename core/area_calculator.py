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

    def calculate_area(self, mask: np.ndarray, return_mask: bool = True) -> Dict[str, Any]:
        total_pixels = mask.size
        processed_pixels = int(np.sum(mask))

        # Процент обработанной площади
        processed_percentage = (processed_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        # Конвертация в квадратные метры
        total_area_sqm = self.calibration.pixels_to_square_meters(processed_pixels)

        logger.info(f"Processed pixels: {processed_pixels}/{total_pixels} "
                    f"({processed_percentage:.2f}%) = {total_area_sqm} sqm")

        result = {
            'total_area_sqm': total_area_sqm,
            'processed_percentage': round(processed_percentage, 2),
            'processed_pixels': processed_pixels,
            'total_pixels': total_pixels
        }

        result['mask'] = self._encode_mask_to_base64(mask)

        return result

    def calculate_area_with_confidence(self, mask: np.ndarray,
                                       confidence_threshold: float = 0.7,
                                       return_mask: bool = False) -> Dict[str, Any]:
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            binary_mask = (mask >= confidence_threshold).astype(np.uint8)
        else:
            binary_mask = mask

        return self.calculate_area(binary_mask, return_mask)

    def _encode_mask_to_base64(self, mask: np.ndarray) -> str:
        """Кодирует бинарную маску в base64 строку"""
        # Преобразуем маску в изображение (0-255)
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Кодируем в PNG
        _, buffer = cv2.imencode('.png', mask_uint8)

        # Конвертируем в base64
        mask_base64 = base64.b64encode(buffer).decode('utf-8')

        return f"data:image/png;base64,{mask_base64}"