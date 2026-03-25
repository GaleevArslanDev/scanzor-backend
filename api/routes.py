from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
import json
import logging

from .schemas import AnalysisResponse, HealthResponse, CalibrationData, TaskType
from core.image_processor import ImageProcessor
from utils.validators import ImageValidator, CalibrationValidator

logger = logging.getLogger(__name__)

router = APIRouter()

# Инициализация процессора
image_processor = ImageProcessor()


@router.post('/analyze/image', response_model=AnalysisResponse)
async def analyze_image(
        file: UploadFile = File(...),
        calibration: str = Form(...),
        task_type: Optional[TaskType] = Form(None, description="Тип задачи: snow, grass или auto (автоопределение)")
):
    """
    Анализ изображения с калибровкой

    Args:
        file: Загруженное изображение
        calibration: JSON строка с данными калибровки
    """
    try:
        # 1. Чтение данных калибровки
        try:
            calibration_data = json.loads(calibration)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in calibration: {str(e)}")

        # 2. Валидация калибровки
        is_valid, error_msg = CalibrationValidator.validate(calibration_data)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid calibration: {error_msg}")

        # 3. Чтение файла
        file_bytes = await file.read()

        # 4. Валидация изображения
        is_valid, error_msg = ImageValidator.validate_image(file_bytes, file.filename)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid image: {error_msg}")

        # 5. Обработка изображения
        task_type_value = task_type.value if task_type else "auto"
        result = image_processor.process_image(file_bytes, calibration_data, task_type_value)

        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result.get('error', 'Processing failed'))

        # 6. Формирование ответа
        return AnalysisResponse(
            status=result['status'],
            total_area_sqm=result['total_area_sqm'],
            processed_percentage=result['processed_percentage'],
            processed_pixels=result['processed_pixels'],
            total_pixels=result['total_pixels'],
            calibration_used=CalibrationData(**result['calibration_used']),
            image_dimensions=result['image_dimensions'],
            mask=result.get('mask'),
            task_type=result.get('task_type')
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in analyze_image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get('/health', response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status='healthy',
        version='1.0.0',
        service='Scanzor API'
    )