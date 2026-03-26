import math
from app.config import (
    FIT_TOLERANCE, 
    CHEST_WEIGHT, WAIST_WEIGHT, HIP_WEIGHT,
    SHOULDER_WEIGHT, HEIGHT_WEIGHT, WEIGHT_WEIGHT
)

def score_measurement(value, min_val, max_val, tolerance=FIT_TOLERANCE):
    if min_val <= value <= max_val:
        return 1.0

    if value < min_val:
        diff = min_val - value
        sigma = tolerance / 1.5
    else:
        diff = value - max_val
        sigma = tolerance

    score = math.exp(-(diff**2) / (2 * (sigma**2)))
    
    return max(0.0, round(score, 3))

def get_weights(product_type):
    p_type = str(product_type).strip().lower()

    if p_type == "top":
        return {"chest": 0.6, "waist": 0.2, "hip": 0.2}
    elif p_type == "bottom":
        return {"chest": 0.1, "waist": 0.5, "hip": 0.4}
    else:
        return {"chest": CHEST_WEIGHT, "waist": WAIST_WEIGHT, "hip": HIP_WEIGHT}

def calculate_fit_score(data):
    chest_score = score_measurement(data.chest, data.size_chest_min, data.size_chest_max)
    waist_score = score_measurement(data.waist, data.size_waist_min, data.size_waist_max)
    hip_score = score_measurement(data.hip, data.size_hip_min, data.size_hip_max)

    weights = get_weights(data.product_type)

    total_weight = sum(weights.values())
    
    fit_score = (
        chest_score * weights["chest"] +
        waist_score * weights["waist"] +
        hip_score * weights["hip"]
    ) / total_weight if total_weight > 0 else 0

    return round(fit_score, 3)