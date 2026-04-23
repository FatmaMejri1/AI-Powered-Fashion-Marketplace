from pydantic import BaseModel
from typing import List, Optional

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

class ProductTarget(BaseModel):
    id: str
    category: str
    product_type: str
    brand: str
    fit_score: float
    similarity_score: float
    description: Optional[str] = None

class RecommendRequest(BaseModel):
    user_id: int
    gender: str
    total_orders: int
    last_category: Optional[str] = "casual"
    last_product_type: Optional[str] = "t-shirt"
    days_since_last_order: Optional[int] = 30
    search_query: Optional[str] = None
    user_measurements: Optional[dict] = None
    products: List[ProductTarget]