import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AreaSegmentation:
    def __init__(self, method: str = "color_threshold"):
        """
        Инициализация сегментации

        Args:
            method: Метод сегментации ('color_threshold' или 'edge_detection')
        """
        self.method = method

    def segment(self, image: np.ndarray, task_type: str = "snow") -> np.ndarray:
        """
        Сегментация изображения для выделения обработанной зоны

        Args:
            image: Изображение в формате numpy array (BGR)
            task_type: Тип задачи ('snow' - снег, 'grass' - трава)

        Returns:
            Бинарная маска (1 - обработанная зона, 0 - необработанная)
        """
        if self.method == "color_threshold":
            return self._color_threshold_segmentation(image, task_type)
        elif self.method == "edge_detection":
            return self._edge_detection_segmentation(image)
        else:
            return self._simple_threshold_segmentation(image, task_type)

    def _color_threshold_segmentation(self, image: np.ndarray, task_type: str) -> np.ndarray:
        """
        Сегментация на основе цветовых порогов

        Для снега: ищем светлые области (очищенный асфальт vs снег)
        Для травы: ищем различия в зеленом канале
        """
        # Конвертация в HSV для лучшего выделения
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if task_type == "snow":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Определяем, какой цвет (светлый или темный) является "снегом"
            # Снег должен быть светлым (высокое значение яркости)
            # Находим среднюю яркость светлых и темных областей
            mean_white = np.mean(gray[mask == 255]) if np.any(mask == 255) else 0
            mean_black = np.mean(gray[mask == 0]) if np.any(mask == 0) else 0

            # Если средняя яркость темных пикселей выше, чем светлых - значит Otsu инвертировал
            if mean_black > mean_white:
                # Инвертируем маску, чтобы снег был белым
                mask = cv2.bitwise_not(mask)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        elif task_type == "grass":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)

            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        mask = (mask > 0).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = image.shape[0] * image.shape[1] * 0.01
        mask_filtered = np.zeros_like(mask)

        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(mask_filtered, [contour], -1, 1, -1)

        return mask_filtered

    def _edge_detection_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Сегментация через обнаружение границ"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blurred, 50, 150)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return (mask > 0).astype(np.uint8)

    def _simple_threshold_segmentation(self, image: np.ndarray, task_type: str) -> np.ndarray:
        """Простая пороговая сегментация"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if task_type == "snow":
            _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        else:
            _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        return (mask > 0).astype(np.uint8)


class SegmentationError(Exception):
    # десь можно добавить ошиьку
    pass