from typing import Optional, Tuple
import io
import logging
import imghdr

logger = logging.getLogger(__name__)


class ImageValidator:
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
    ALLOWED_MIME_TYPES = {
        'image/jpeg',
        'image/jpg',
        'image/png'
    }

    MAX_FILE_SIZE_MB = 10  # Максимальный размер файла в МБ

    @classmethod
    def validate_image(cls, file_bytes: bytes, filename: str) -> Tuple[bool, Optional[str]]:
        """
        Валидация изображения

        Args:
            file_bytes: Байты файла
            filename: Имя файла

        Returns:
            (is_valid, error_message)
        """
        # Проверка размера
        file_size_mb = len(file_bytes) / (1024 * 1024)
        if file_size_mb > cls.MAX_FILE_SIZE_MB:
            return False, f"File size exceeds {cls.MAX_FILE_SIZE_MB} MB"

        # Проверка расширения файла
        import os
        ext = os.path.splitext(filename)[1].lower()
        if ext not in cls.ALLOWED_EXTENSIONS:
            return False, f"File extension {ext} not allowed. Allowed: {cls.ALLOWED_EXTENSIONS}"

        # Проверка содержимого файла (магические числа)
        try:
            # Используем imghdr для определения типа изображения
            image_type = imghdr.what(None, h=file_bytes)
            if image_type not in ['jpeg', 'png']:
                return False, f"File content type {image_type} not allowed. Allowed: jpeg, png"
        except Exception as e:
            logger.warning(f"Could not detect image type: {e}")
            # Если не можем определить, пропускаем проверку
            pass

        return True, None


class CalibrationValidator:
    """Валидатор данных калибровки"""

    @classmethod
    def validate(cls, calibration_data: dict) -> Tuple[bool, Optional[str]]:
        """
        Валидация данных калибровки

        Args:
            calibration_data: Данные калибровки

        Returns:
            (is_valid, error_message)
        """
        required_fields = ['focalLength', 'mountHeight', 'tiltAngle', 'sensorWidth', 'imageWidth']

        # Проверка наличия всех полей
        for field in required_fields:
            if field not in calibration_data:
                return False, f"Missing required field: {field}"

        # Проверка типов и значений
        try:
            if calibration_data['focalLength'] <= 0:
                return False, "focalLength must be positive"

            if calibration_data['mountHeight'] <= 0:
                return False, "mountHeight must be positive"

            if not 0 <= calibration_data['tiltAngle'] <= 90:
                return False, "tiltAngle must be between 0 and 90"

            if calibration_data['sensorWidth'] <= 0:
                return False, "sensorWidth must be positive"

            if calibration_data['imageWidth'] <= 0:
                return False, "imageWidth must be positive"

        except (TypeError, ValueError) as e:
            return False, f"Invalid calibration data type: {str(e)}"

        return True, None