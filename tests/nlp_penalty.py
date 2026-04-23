import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_nlp_penalty():
    # We choose p004 because our mock REVIEWS in review_analyzer.py 
    # has multiple complaints about p004 being "tight on shoulders".
    # Therefore, a user with broad shoulders (e.g. 50cm > 44cm threshold) 
    # should receive a high risk_penalty.
    
    payload = {
        "user_id": 42,
        "gender": "M",
        "total_orders": 15,
        "last_category": "formal",
        "last_product_type": "blazer",
        "days_since_last_order": 5,
        "search_query": "formal tailored blazer",
        
        # This is the new field we added to the schema!
        "user_measurements": {
            "shoulder": 50,  # Broad shoulder -> trigger penalty
            "chest": 95,
            "waist": 82,
            "hips": 95
        },
        "products": [
            {
                "id": "p004",  # The Gucci Blazer with bad shoulder reviews
                "category": "formal",
                "product_type": "blazer",
                "brand": "Gucci",
                "fit_score": 0.90,  # Good baseline mathematical fit
                "similarity_score": 0.80, # Body twin likes it
                "description": "Slim fit wool blazer with notched lapels and structured shoulders."
            },
            {
                "id": "p003",  # H&M tee, mostly good reviews
                "category": "casual",
                "product_type": "t-shirt",
                "brand": "H&M",
                "fit_score": 0.90,
                "similarity_score": 0.80,
                "description": "Regular fit crew neck t-shirt in soft jersey fabric."
            }
        ]
    }
    
    try:
        print("Sending recommendation request...")
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/recommend", json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            results = response.json()
            print(f"\nRecommendation Results (Took {(end_time - start_time):.2f}s):")
            
            for rec in results['recommendations']:
                risk = rec.get("risk_data", {})
                penalty = rec.get("return_penalty", 0.0)
                prob = rec.get("return_probability", 0.0)
                final = rec.get("final_score", 0.0)
                
                print(f"\n--- Product: {rec['product_id']} ---")
                print(f"  Final Score:       {final:.3f}")
                print(f"  Return Prob:       {prob:.4f}")
                print(f"  Applied Penalty:  -{penalty:.3f}")
                
                if risk:
                    print(f"  Product Fit Risk:  {risk.get('fit_risk_score')}")
                    print(f"  Top Complaint:     {risk.get('dominant_complaint')}")
                    print(f"  Complaint Details: {risk.get('complaint_breakdown')}")
                else:
                    print("  No risk data found.")
                    
        else:
            print(f"Recommendation failed (Status: {response.status_code})")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("ERROR: Connection Refused. Make sure you have started the server:")
        print("       python -m uvicorn app.main:app --reload")

if __name__ == "__main__":
    print("--- STARTING NLP PENALTY VALIDATION ---")
    test_nlp_penalty()
    print("--- VALIDATION COMPLETE ---")
