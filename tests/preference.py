import os
import sys

# Ensure the root directory is accessible so 'app' can be imported correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.preference_service import PreferenceService

def test_preference_service():
    print("--- STARTING PREFERENCE SERVICE VALIDATION ---\n")
    
    # 1. Initialize Service
    try:
        print("[1] Initializing PreferenceService...")
        service = PreferenceService()
        if not service.pipeline:
            print("❌ WARNING: ML Pipeline failed to load. Is preference_model.pkl missing?")
        else:
            print("✅ ML Pipeline loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize service: {e}")
        return

    # 2. Test Build User Context (Returning User vs Cold Start)
    print("\n[2] Testing User Context Builder...")
    # Using User ID 42 (assuming it's in the generated mock training memory)
    returning_user_ctx = service.build_user_context(
        user_id=42, gender="M", purchase_frequency=15, 
        last_category="sportswear", last_product_type="sneakers", days_since=3
    )
    
    if not returning_user_ctx["_cold_start"]:
        print(f"✅ Successfully identified Returning User (ID 42).")
        print(f"    ↳ Favorite Category: {returning_user_ctx.get('fav_category')}")
        print(f"    ↳ Buy Rate: {returning_user_ctx.get('buy_rate')}")
    else:
        print(f"⚠️ User 42 treated as Cold Start. (This is normal if user_memory.pkl was freshly randomized without ID 42).")

    cold_user_ctx = service.build_user_context(
        user_id=99999, gender="F", purchase_frequency=0, 
        last_category="unknown", last_product_type="unknown", days_since=0
    )
    if cold_user_ctx["_cold_start"]:
        print(f"✅ Successfully identified Cold Start User (ID 99999).")

    # 3. Test Preference Prediction
    print("\n[3] Testing ML Preference Prediction...")
    mock_product = {
        "category": "streetwear",
        "product_type": "hoodie",
        "brand": "Nike"
    }
    
    pref_result = service.predict_preference(returning_user_ctx, mock_product)
    print(f"✅ Prediction for {mock_product['brand']} {mock_product['product_type']}:")
    print(f"    ↳ Score: {pref_result['preference_score']}")
    print(f"    ↳ Label: {pref_result['label']}")

    # 4. Test Popularity Fallback (Cold Start)
    print("\n[4] Testing Popularity Cache (Cold Start Fallback)...")
    pop_score = service.get_popularity_score("streetwear", "hoodie", gender="F")
    print(f"✅ Popularity Score (Female, Streetwear, Hoodie): {pop_score}")

    print("\n--- VALIDATION COMPLETE ---")

if __name__ == "__main__":
    test_preference_service()
