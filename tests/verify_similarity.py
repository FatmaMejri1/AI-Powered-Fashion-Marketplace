import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_similarity(payload, label):
    try:
        response = requests.post(f"{BASE_URL}/similar-users", json=payload)
        if response.status_code == 200:
            results = response.json().get('similar_users', [])
            print(f"✅ {label}: Found {len(results)} matches")
            for res in results:
                print(f"   - Match ID: {res['id']} | Score: {res['similarity_score']} | Distance: {res['distance']}")
        else:
            print(f"❌ {label} failed (Status: {response.status_code})")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ {label} failed with error: {e}")

print("--- STARTING SIMILARITY VALIDATION ---")

test_similarity({
  "product_type": "top",
  "gender": "male",
  "height": 178, "weight": 72,
  "chest": 98, "waist": 82, "hip": 92, "shoulder": 43,
  "size_chest_min": 0, "size_chest_max": 0,
  "size_waist_min": 0, "size_waist_max": 0,
  "size_hip_min": 0, "size_hip_max": 0
}, "Male Similarity Test")

test_similarity({
  "product_type": "top",
  "gender": "female",
  "height": 164, "weight": 58,
  "chest": 88, "waist": 72, "hip": 90, "shoulder": 39,
  "size_chest_min": 0, "size_chest_max": 0,
  "size_waist_min": 0, "size_waist_max": 0,
  "size_hip_min": 0, "size_hip_max": 0
}, "Female Similarity Test")

print("--- VALIDATION COMPLETE ---")
