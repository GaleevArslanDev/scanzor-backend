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

    def process_image(self, image_bytes: bytes, calibration_data: Dict[str, Any],
                      task_type: str = "auto") -> Dict[str, Any]:
        """
        Полная обработка изображения

        Args:
            image_bytes: Байты изображения
            calibration_data: Данные калибровки
            task_type: Тип задачи ('snow', 'grass', 'auto')

        Returns:
            Результаты анализа
        """
        try:
            # 1. Загрузка изображения
            image = self._load_image(image_bytes)
            image_height, image_width = image.shape[:2]

            logger.info(f"Loaded image: {image_width}x{image_height}")

            # 2. Калибровка с реальными размерами изображения
            calibration = CameraCalibration(calibration_data, (image_height, image_width))

            # 3. Определение типа задачи
            if task_type == "auto":
                actual_task_type = self._detect_task_type(image)
                logger.info(f"Auto-detected task type: {actual_task_type}")
            else:
                actual_task_type = task_type
                logger.info(f"Using specified task type: {actual_task_type}")

            # 4. Сегментация
            mask = self.segmenter.segment(image, actual_task_type)

            # Убеждаемся, что маска имеет правильный размер
            if mask.shape != (image_height, image_width):
                mask = cv2.resize(mask.astype(np.float32),
                                  (image_width, image_height),
                                  interpolation=cv2.INTER_NEAREST)
                mask = mask > 0.5

            # 5. Расчет площади
            calculator = AreaCalculator(calibration)
            area_results = calculator.calculate_area(mask)

            # 6. Формирование результата
            result = {
                'status': 'success',
                **area_results,
                'calibration_used': calibration_data,
                'image_dimensions': {
                    'width': image_width,
                    'height': image_height
                },
                'task_type': actual_task_type
            }

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