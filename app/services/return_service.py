import pickle
import pandas as pd
import os
from app.config import RETURN_RISK_FEATURES

MODEL_PATH = os.path.join("app", "ml_models", "return_risk_model.pkl")

class ReturnService:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)

    def predict_return_probability(self, features: dict) -> float:
        if self.model is None:
            return 0.15

        df = pd.DataFrame([features])[RETURN_RISK_FEATURES]
        prob = self.model.predict_proba(df)[0][1]
        return round(float(prob), 4)

    def calculate_risk_penalty(self, prob: float) -> float:
        if prob < 0.3:
            return 0.0
        
        penalty = (prob ** 2) * 0.5 
        return round(penalty, 4)
