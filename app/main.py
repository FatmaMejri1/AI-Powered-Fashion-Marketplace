import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.models.schema import FitRequest, RecommendRequest
from app.services.fit_service import calculate_fit_score
from app.services.similarity import SimilarityService
from app.services.preference_service import PreferenceService
from app.services.return_service import ReturnService
from app.services.ranking_service import RankingService
from app.data.mock_users import USERS_DB
from app.nlp.description_parser import get_sentence_embeddings, enrich_product_for_recommender
from app.nlp.review_analyzer import get_nlp_penalty_for_ranking
from app.data.mock_reviews import REVIEWS
from app.nlp.model_loader import get_shared_model
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="AI Recommendation Service")
preference_service = PreferenceService()
return_service = ReturnService()
ranking_service = RankingService()

@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "trace": traceback.format_exc()}
    )

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

    texts_to_embed = []
    if data.search_query:
        texts_to_embed.append(data.search_query)
    for prod in data.products:
        if prod.description:
            texts_to_embed.append(prod.description)

    embeddings_map = {}
    if texts_to_embed:
        embeddings = get_sentence_embeddings(texts_to_embed)
        for i, text in enumerate(texts_to_embed):
            embeddings_map[text] = embeddings[i]

    results_buffer = []
    for prod in data.products:
        enriched_features = {}
        if prod.description:
            enriched_features = enrich_product_for_recommender(
                prod.id, prod.description, None
            )

        nlp_score = 0.0
        if data.search_query and prod.description:
            query_emb = embeddings_map.get(data.search_query)
            prod_emb  = embeddings_map.get(prod.description)
            if query_emb is not None and prod_emb is not None:
                nlp_score = max(0.0, float(
                    cosine_similarity([query_emb], [prod_emb])[0][0]
                ))

        if user_ctx["_cold_start"]:
            pref_score = preference_service.get_popularity_score(
                prod.category, prod.product_type, gender=data.gender
            )
        else:
            pref = preference_service.predict_preference(
                user_ctx, prod.model_dump()
            )
            pref_score = pref["preference_score"]

        fit_risk_penalty = 0.0
        risk_data = None
        if data.user_measurements:
            shared_model = get_shared_model()
            risk_data = get_nlp_penalty_for_ranking(
                user_measurements=data.user_measurements,
                product_id=prod.id,
                all_reviews=REVIEWS,
                model=shared_model
            )
            fit_risk_penalty = risk_data["personalized_penalty"]

        risk_features = {
            "fit_score":                prod.fit_score,
            "size_mismatch":            int(prod.fit_score < 0.6),
            "morpho_similarity_score":  prod.similarity_score,
            "preference_score":         pref_score,
            "fit_risk_score":           risk_data["fit_risk_score"] if risk_data else 0.1,
            "personalized_nlp_penalty": fit_risk_penalty,
            "historical_return_rate":   0.20,
            "size_inconsistency":       0.0,
            "user_return_tendency":     0.15,
            "body_mismatch_score":      fit_risk_penalty 
        }

        return_prob = return_service.predict_return_probability(risk_features)
        return_penalty = return_service.calculate_risk_penalty(return_prob)

        results_buffer.append({
            "product_id":               prod.id,
            "fit_score":                prod.fit_score,
            "preference_score":         pref_score,
            "similarity_score":         prod.similarity_score,
            "nlp_score":                nlp_score,
            "return_penalty":           return_penalty,
            "return_probability":       return_prob,
            "cold_start":               user_ctx["_cold_start"],
            "risk_data":                risk_data,
        })

    ranked = ranking_service.rank_batch(results_buffer)
    return {"user_id": data.user_id, "recommendations": ranked}