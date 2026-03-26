import pickle
import pandas as pd
import os
from app.config import FIT_WEIGHTS, COLD_START_DEFAULTS

MODEL_PATH = os.path.join("app", "ml_models", "preference_model.pkl")
MEMORY_PATH = os.path.join("app", "ml_models", "user_memory.pkl")
POPULARITY_PATH = os.path.join("app", "ml_models", "popularity_cache.pkl")

class PreferenceService:
    def __init__(self):
        self.pipeline = None
        self.memory = None
        self.popularity_cache = None
        self.load_artifacts()

    def load_artifacts(self):
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                self.pipeline = pickle.load(f)
        if os.path.exists(MEMORY_PATH):
            with open(MEMORY_PATH, "rb") as f:
                self.memory = pickle.load(f)
        if os.path.exists(POPULARITY_PATH):
            with open(POPULARITY_PATH, "rb") as f:
                self.popularity_cache = pickle.load(f)

    def build_user_context(self, user_id, gender, purchase_frequency, last_category, last_product_type, days_since):
        if self.memory is not None:
            row = self.memory[self.memory["user_id"] == user_id]
            if not row.empty:
                r = row.iloc[0]
                return {
                    "gender": gender,
                    "purchase_frequency": purchase_frequency,
                    "avg_frequency": float(r["avg_frequency"]),
                    "fav_category": r["fav_category"],
                    "fav_brand": r["fav_brand"],
                    "fav_product_type": r["fav_product_type"],
                    "buy_rate": r["buy_rate"],
                    "last_category": last_category,
                    "last_product_type": last_product_type,
                    "days_since_last_purchase": days_since,
                    "_cold_start": False,
                }
        
        return {
            "gender": gender,
            "purchase_frequency": purchase_frequency,
            **COLD_START_DEFAULTS,
            "last_category": last_category,
            "last_product_type": last_product_type,
            "days_since_last_purchase": days_since,
            "_cold_start": True,
        }

    def predict_preference(self, user_ctx, product):
        if self.pipeline is None:
            return {"preference_score": 0.5, "will_buy": False, "label": "UNKNOWN"}

        feat = {
            "category": product["category"],
            "product_type": product["product_type"],
            "brand": product["brand"],
            "gender": user_ctx["gender"],
            "purchase_frequency": user_ctx["purchase_frequency"],
            "avg_frequency": user_ctx.get("avg_frequency", COLD_START_DEFAULTS["avg_frequency"]),
            "fav_category": user_ctx["fav_category"],
            "fav_brand": user_ctx["fav_brand"],
            "fav_product_type": user_ctx["fav_product_type"],
            "buy_rate": user_ctx["buy_rate"],
            "last_category": user_ctx["last_category"],
            "last_product_type": user_ctx["last_product_type"],
            "days_since_last_purchase": user_ctx["days_since_last_purchase"],
            "is_fav_category": int(product["category"] == user_ctx["fav_category"]),
            "is_fav_brand": int(product["brand"] == user_ctx["fav_brand"]),
        }
        
        prob = self.pipeline.predict_proba(pd.DataFrame([feat]))[0][1]
        return {
            "preference_score": round(float(prob), 4),
            "will_buy": prob >= 0.5,
            "label": "LIKED" if prob >= 0.5 else "DISLIKED",
        }

    def get_popularity_score(self, category, product_type, gender=""):
        if self.popularity_cache is None:
            return 0.5

        if gender:
            score = self.popularity_cache.get((gender, category, product_type))
            if score is not None:
                return score
        return self.popularity_cache.get((category, product_type), 0.50)

    def preference_mismatch_penalty(self, user_ctx, product):
        penalty = 0.0
        if product.get("category", "") != user_ctx.get("fav_category", ""):
            penalty += 0.05
        if product.get("product_type", "") != user_ctx.get("fav_product_type", ""):
            penalty += 0.03
        return round(penalty, 4)

    def compute_final_score(self, fit_score, preference_score, similarity_score, product_type, user_ctx=None, product=None):
        w_fit = FIT_WEIGHTS.get(product_type, 0.50)
        rem = 1.0 - w_fit
        w_pref = round(rem * 0.45, 4)
        w_sim = round(rem * 0.25, 4)
        w_nlp = round(rem * 0.30, 4)
        w_risk = 0.15

        mismatch = self.preference_mismatch_penalty(user_ctx, product) if user_ctx and product else 0.0

        score = (w_fit * fit_score
                 + w_pref * preference_score
                 + w_sim * similarity_score
                 - w_risk * 0.0 # return_risk placeholder
                 - mismatch)

        return {
            "final_score": round(score, 4),
            "weights": {
                "fit": w_fit,
                "preference": w_pref,
                "similarity": w_sim,
                "risk_penalty": w_risk,
            },
        }
