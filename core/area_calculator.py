import numpy as np
from typing import Dict, Any
from .calibration import CameraCalibration
import logging

logger = logging.getLogger(__name__)


class AreaCalculator:
    def __init__(self, calibration: CameraCalibration):
        self.calibration = calibration

    def calculate_area(self, mask: np.ndarray) -> Dict[str, Any]:
        total_pixels = mask.size
        processed_pixels = int(np.sum(mask))

        # Процент обработанной площади
        processed_percentage = (processed_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        # Конвертация в квадратные метры
        total_area_sqm = self.calibration.pixels_to_square_meters(processed_pixels)

        logger.info(f"Processed pixels: {processed_pixels}/{total_pixels} "
                    f"({processed_percentage:.2f}%) = {total_area_sqm} sqm")

        return {
            'total_area_sqm': total_area_sqm,
            'processed_percentage': round(processed_percentage, 2),
            'processed_pixels': processed_pixels,
            'total_pixels': total_pixels
        }

    def calculate_area_with_confidence(self, mask: np.ndarray,
                                       confidence_threshold: float = 0.7) -> Dict[str, Any]:
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            binary_mask = (mask >= confidence_threshold).astype(np.uint8)
        else:
            binary_mask = mask

        return self.calculate_area(binary_mask)