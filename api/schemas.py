from pydantic import BaseModel, Field, validator
from typing import Optional
from enum import Enum


class CalibrationData(BaseModel):
    focalLength: float = Field(..., gt=0, description="Фокусное расстояние в мм")
    mountHeight: float = Field(..., gt=0, description="Высота подвеса в метрах")
    tiltAngle: float = Field(..., ge=0, le=90, description="Угол наклона в градусах")
    sensorWidth: float = Field(..., gt=0, description="Ширина сенсора в мм")
    imageWidth: int = Field(..., gt=0, description="Ширина изображения в пикселях")

    @validator('tiltAngle')
    def validate_tilt_angle(cls, v):
        if v < 0 or v > 90:
            raise ValueError('tiltAngle должен быть между 0 и 90 градусами')
        return v


class AnalysisResponse(BaseModel):
    status: str
    total_area_sqm: float
    processed_percentage: float
    processed_pixels: int
    total_pixels: int
    calibration_used: CalibrationData
    image_dimensions: dict
    mask: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    service: str