from fastapi import FastAPI
from app.models.schema import FitRequest, RecommendRequest
from app.services.fit_service import calculate_fit_score
from app.services.similarity import SimilarityService
from app.services.preference_service import PreferenceService
from app.data.mock_users import USERS_DB

app = FastAPI(title="AI Recommendation Service")
preference_service = PreferenceService()

@app.get("/")
def root():
    return {"message": "AI Service Running"}

@app.post("/fit-score")
def fit_score(data: FitRequest):
    score = calculate_fit_score(data)
    return {"fit_score": score}

@app.post("/similar-users")
def get_similar_users(data: FitRequest):
    service = SimilarityService(USERS_DB)
    similar_ids = service.find_similar_users(data, k=3)
    return {"similar_users": similar_ids}

@app.post("/recommend")
def recommend(data: RecommendRequest):
    user_ctx = preference_service.build_user_context(
        user_id=data.user_id,
        gender=data.gender,
        purchase_frequency=data.total_orders,
        last_category=data.last_category,
        last_product_type=data.last_product_type,
        days_since=data.days_since_last_order
    )
    
    ranked = []
    for prod in data.products:
        if user_ctx["_cold_start"]:
            pref_score = preference_service.get_popularity_score(
                prod.category, 
                prod.product_type, 
                gender=data.gender
            )
        else:
            pref = preference_service.predict_preference(user_ctx, prod.dict())
            pref_score = pref["preference_score"]
            
        final = preference_service.compute_final_score(
            fit_score=prod.fit_score,
            preference_score=pref_score,
            similarity_score=prod.similarity_score,
            product_type=prod.product_type,
            user_ctx=user_ctx,
            product=prod.dict()
        )
        
        ranked.append({
            "product_id": prod.id,
            "final_score": final["final_score"],
            "preference_score": pref_score,
            "weights": final["weights"],
            "cold_start": user_ctx["_cold_start"]
        })
        
    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return {"user_id": data.user_id, "recommendations": ranked}