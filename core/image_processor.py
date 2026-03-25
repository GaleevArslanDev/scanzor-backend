import cv2
import numpy as np
from typing import Dict, Any, Tuple
from PIL import Image
import io
import logging

from .calibration import CameraCalibration, CalibrationError
from .segmentation import AreaSegmentation, SegmentationError
from .area_calculator import AreaCalculator

logger = logging.getLogger(__name__)


class ImageProcessor:
    def __init__(self):
        self.segmenter = AreaSegmentation(method="color_threshold")

    def process_image(self, image_bytes: bytes, calibration_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Полная обработка изображения

        Args:
            image_bytes: Байты изображения
            calibration_data: Данные калибровки

        Returns:
            Результаты анализа
        """
        try:
            # 1. Загрузка изображения
            image = self._load_image(image_bytes)

            # 2. Калибровка
            calibration = CameraCalibration(calibration_data)

            # 3. Определение типа задачи (можно добавить автоматическое определение)
            if task_type == "auto":
                actual_task_type = self._detect_task_type(image)
                logger.info(f"Auto-detected task type: {actual_task_type}")
            else:
                actual_task_type = task_type
                logger.info(f"Using specified task type: {actual_task_type}")

            # 4. Сегментация
            mask = self.segmenter.segment(image, task_type)

            # 5. Расчет площади
            calculator = AreaCalculator(calibration)
            area_results = calculator.calculate_area(mask)

            # 6. Формирование результата
            result = {
                'status': 'success',
                **area_results,
                'calibration_used': calibration_data,
                'image_dimensions': {
                    'width': image.shape[1],
                    'height': image.shape[0]
                },
                'task_type': task_type
            }

            # Для отладки: сохранить маску (опционально)
            # self._save_mask_for_debug(mask)

            return result

        except (CalibrationError, SegmentationError) as e:
            logger.error(f"Processing error: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {
                'status': 'error',
                'error': f'Internal processing error: {str(e)}'
            }

    def _load_image(self, image_bytes: bytes) -> np.ndarray:
        pil_image = Image.open(io.BytesIO(image_bytes))

        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        image = np.array(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image

    def _detect_task_type(self, image: np.ndarray) -> str:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        avg_saturation = np.mean(hsv[:, :, 1])

        avg_value = np.mean(hsv[:, :, 2])

        if avg_saturation < 50 and avg_value > 150:
            return 'snow'
        else:
            return 'grass'

    def _save_mask_for_debug(self, mask: np.ndarray, filename: str = "debug_mask.png"):
        mask_visual = (mask * 255).astype(np.uint8)
        cv2.imwrite(filename, mask_visual)