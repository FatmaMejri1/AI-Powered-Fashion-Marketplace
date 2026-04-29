import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_recommendation():
    payload = {
        "user_id": 42,
        "gender": "M",
        "total_orders": 15,
        "last_category": "sportswear",
        "last_product_type": "sneakers",
        "days_since_last_order": 3,
        "search_query": "stretchy flexible clothing that is formal",
        "products": [
            {
                "id": "p1",
                "category": "streetwear",
                "product_type": "hoodie",
                "brand": "Nike",
                "fit_score": 0.92,
                "similarity_score": 0.80,
                "description": "Oversized fit cotton hoodie with ribbed cuffs and kangaroo pocket. Relaxed silhouette perfect for streetwear layering. 100% heavyweight cotton."
            },
            {
                "id": "p2",
                "category": "formal",
                "product_type": "jacket",
                "brand": "Zara",
                "fit_score": 0.85,
                "similarity_score": 0.50,
                "description": "Slim fit blazer in stretch fabric. Tailored silhouette perfectly flexible for formal events."
            }
        ]
    }
    
    try:
        response = client.post("/recommend", json=payload)
        if response.status_code == 200:
            results = response.json()
            print("Recommendation Results:")
            for rec in results['recommendations']:
                print(f"   - {rec['product_id']} | Final: {rec['final_score']} | NLP: {rec.get('nlp_score', 0):.3f} | Pref: {rec['preference_score']}")
        else:
            print(f"Recommendation failed (Status: {response.status_code})")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Test failed with error: {e}")

if __name__ == "__main__":
    print("--- STARTING RECOMMENDATION VALIDATION ---")
    test_recommendation()
    print("--- VALIDATION COMPLETE ---")
