from fastapi import FastAPI
from app.models.schema import FitRequest
from app.services.fit_service import calculate_fit_score
from app.services.similarity import SimilarityService
from app.data.mock_users import USERS_DB

app = FastAPI(title="AI Recommendation Service")

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