from pydantic import BaseModel

class FitRequest(BaseModel):
    product_type: str
    gender: str
    height: float
    weight: float
    chest: float
    waist: float
    hip: float
    shoulder: float

    size_chest_min: float
    size_chest_max: float

    size_waist_min: float
    size_waist_max: float

    size_hip_min: float
    size_hip_max: float