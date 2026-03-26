import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_recommendation():
    payload = {
        "user_id": 42,
        "gender": "M",
        "total_orders": 15,
        "last_category": "sportswear",
        "last_product_type": "sneakers",
        "days_since_last_order": 3,
        "products": [
            {
                "id": "p1",
                "category": "streetwear",
                "product_type": "hoodie",
                "brand": "Nike",
                "fit_score": 0.92,
                "similarity_score": 0.80
            },
            {
                "id": "p2",
                "category": "formal",
                "product_type": "jacket",
                "brand": "Zara",
                "fit_score": 0.85,
                "similarity_score": 0.50
            }
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/recommend", json=payload)
        if response.status_code == 200:
            results = response.json()
            print("Recommendation Results:")
            for rec in results['recommendations']:
                print(f"   - Product: {rec['product_id']} | Final Score: {rec['final_score']} | Preference: {rec['preference_score']}")
        else:
            print(f"Recommendation failed (Status: {response.status_code})")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Test failed with error: {e}")

if __name__ == "__main__":
    print("--- STARTING RECOMMENDATION VALIDATION ---")
    test_recommendation()
    print("--- VALIDATION COMPLETE ---")
