import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_fit_score(payload, label):
    try:
        response = requests.post(f"{BASE_URL}/fit-score", json=payload)
        if response.status_code == 200:
            print(f"✅ {label}: {response.json()['fit_score']}")
        else:
            print(f"❌ {label} failed (Status: {response.status_code})")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ {label} failed with error: {e}")

print("--- STARTING FIT SERVICE VALIDATION ---")

test_fit_score({
    "product_type": "top",
    "height": 180, "weight": 75,
    "chest": 98, "waist": 85, "hip": 95, "shoulder": 45,
    "size_chest_min": 95, "size_chest_max": 105,
    "size_waist_min": 80, "size_waist_max": 95,
    "size_hip_min": 85, "size_hip_max": 105
}, "Perfect Top Fit")

test_fit_score({
    "product_type": "top",
    "height": 180, "weight": 75,
    "chest": 108, "waist": 85, "hip": 95, "shoulder": 45,
    "size_chest_min": 95, "size_chest_max": 105,
    "size_waist_min": 80, "size_waist_max": 95,
    "size_hip_min": 85, "size_hip_max": 105
}, "Tight Top Fit")

test_fit_score({
    "product_type": "bottom",
    "height": 180, "weight": 75,
    "chest": 120, "waist": 82, "hip": 92, "shoulder": 45,
    "size_chest_min": 90, "size_chest_max": 100,
    "size_waist_min": 80, "size_waist_max": 85,
    "size_hip_min": 90, "size_hip_max": 95
}, "Perfect Bottom Fit")

test_fit_score({
    "product_type": "bottom",
    "height": 180, "weight": 75,
    "chest": 95, "waist": 90, "hip": 92, "shoulder": 45,
    "size_chest_min": 90, "size_chest_max": 100,
    "size_waist_min": 80, "size_waist_max": 85,
    "size_hip_min": 90, "size_hip_max": 95
}, "Loose Bottom Fit")

print("--- VALIDATION COMPLETE ---")
