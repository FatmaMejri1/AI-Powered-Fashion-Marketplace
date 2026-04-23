import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

from app.config import (
    CATEGORIES, FIT_TYPES, SIZES, SIZE_CHEST_MAP,
    CATEGORY_RETURN_BASE, RETURN_RISK_FEATURES, COLD_START_DEFAULTS
)

np.random.seed(42)

def simulate_users(n: int) -> pd.DataFrame:
    users = pd.DataFrame({
        "user_id": [f"u{str(i).zfill(4)}" for i in range(n)],
        "height":   np.random.normal(170, 12, n).clip(150, 210),
        "weight":   np.random.normal(70,  15, n).clip(45,  130),
        "chest":    np.random.normal(95,  12, n).clip(70,  130),
        "waist":    np.random.normal(80,  12, n).clip(55,  120),
        "hip":      np.random.normal(97,  12, n).clip(70,  130),
        "shoulder": np.random.normal(42,   5, n).clip(32,   58),
        "user_return_tendency": np.random.beta(2, 5, n),
    })
    return users.round(1)

def simulate_products(n: int) -> pd.DataFrame:
    fit_types  = np.random.choice(FIT_TYPES, n)
    categories = np.random.choice(CATEGORIES, n)
    base_return = np.array([
        (CATEGORY_RETURN_BASE.get(f, 0.20) + CATEGORY_RETURN_BASE.get(c, 0.20)) / 2
        for f, c in zip(fit_types, categories)
    ])
    size_inconsistency = np.random.normal(0, 0.8, n).clip(-2, 2)
    products = pd.DataFrame({
        "product_id":         [f"p{str(i).zfill(4)}" for i in range(n)],
        "category":           categories,
        "fit_type":           fit_types,
        "historical_return_rate": (base_return + np.random.normal(0, 0.05, n)).clip(0.05, 0.60),
        "size_inconsistency": size_inconsistency.round(2),
        "fit_risk_score":     np.random.beta(2, 5, n).round(3),
    })
    return products

def compute_fit_score(user_chest: float, chosen_size: str) -> float:
    size_chest = SIZE_CHEST_MAP.get(chosen_size, 96)
    diff       = abs(user_chest - size_chest)
    score      = max(0.0, 1.0 - (diff / 16))
    return round(score, 3)

def simulate_orders(users: pd.DataFrame, products: pd.DataFrame, n: int) -> pd.DataFrame:
    u_idx = np.random.randint(0, len(users), n)
    p_idx = np.random.randint(0, len(products), n)
    u = users.iloc[u_idx].reset_index(drop=True)
    p = products.iloc[p_idx].reset_index(drop=True)
    
    size_choices = []
    for chest in u["chest"]:
        closest = min(SIZE_CHEST_MAP, key=lambda s: abs(SIZE_CHEST_MAP[s] - chest))
        idx = SIZES.index(closest)
        roll = np.random.random()
        if roll < 0.70: chosen = closest
        elif roll < 0.85: chosen = SIZES[min(idx + 1, len(SIZES) - 1)]
        else: chosen = SIZES[max(idx - 1, 0)]
        size_choices.append(chosen)
    
    chosen_sizes = np.array(size_choices)
    fit_scores = np.array([compute_fit_score(c, s) for c, s in zip(u["chest"], chosen_sizes)])
    size_mismatch = (fit_scores < 0.5).astype(int)
    shoulder_mismatch = ((u["shoulder"].values > 46) & (p["fit_type"].values == "slim")).astype(float)
    chest_mismatch = ((u["chest"].values > 100) & (p["fit_type"].values == "slim")).astype(float)
    body_mismatch_score = ((shoulder_mismatch + chest_mismatch) / 2).round(3)
    personalized_nlp_penalty = (p["fit_risk_score"].values * body_mismatch_score * np.random.uniform(0.5, 1.5, n)).clip(0, 0.5).round(3)
    
    return_prob = (0.25 * (1 - fit_scores) + 0.20 * size_mismatch + 0.15 * body_mismatch_score + 
                   0.15 * p["historical_return_rate"].values + 0.10 * p["fit_risk_score"].values + 
                   0.10 * personalized_nlp_penalty + 0.05 * u["user_return_tendency"].values + 
                   np.random.normal(0, 0.05, n)).clip(0, 1)
    
    returned = (return_prob > 0.45).astype(int)
    
    df = pd.DataFrame({
        "fit_score": fit_scores,
        "size_mismatch": size_mismatch,
        "morpho_similarity_score": np.random.beta(3, 2, n).round(3),
        "preference_score": np.random.beta(3, 2, n).round(3),
        "fit_risk_score": p["fit_risk_score"].values,
        "personalized_nlp_penalty": personalized_nlp_penalty,
        "historical_return_rate": p["historical_return_rate"].values,
        "size_inconsistency": p["size_inconsistency"].values,
        "user_return_tendency": u["user_return_tendency"].values.round(3),
        "body_mismatch_score": body_mismatch_score,
        "returned": returned
    })
    return df

def train_risk_model(df: pd.DataFrame):
    X = df[RETURN_RISK_FEATURES]
    y = df["returned"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"[Training] Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    return clf

def save_model(model):
    os.makedirs("app/ml_models", exist_ok=True)
    with open("app/ml_models/return_risk_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("[Success] Model saved to app/ml_models/")

if __name__ == "__main__":
    print("--- Return Risk Model Training ---")
    data = simulate_orders(simulate_users(1000), simulate_products(500), 5000)
    model = train_risk_model(data)
    save_model(model)
    print("--- Pipeline Complete ---")