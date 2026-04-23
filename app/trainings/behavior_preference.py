import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from app.config import (
    CATEGORIES, BRANDS, GENDERS, PRODUCT_TYPES,
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, FEATURES,
    COLD_START_DEFAULTS
)

def generate_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    gender_affinity = {
        "M":     {"streetwear":0.80,"casual":0.60,"formal":0.30,"sportswear":0.75,"oversized":0.70,"vintage":0.45},
        "F":     {"streetwear":0.50,"casual":0.75,"formal":0.60,"sportswear":0.50,"oversized":0.55,"vintage":0.60},
        "Other": {"streetwear":0.60,"casual":0.65,"formal":0.40,"sportswear":0.60,"oversized":0.65,"vintage":0.55},
    }
    product_affinity = {"t-shirt":0.70,"hoodie":0.80,"jacket":0.65,"jeans":0.60,"shorts":0.55,"dress":0.50,"sneakers":0.75}
    
    n_users = 300
    user_ids = np.random.randint(1, n_users + 1, size=n_samples)
    user_bias_map = {uid: np.random.normal(0, 0.05) for uid in range(1, n_users + 1)}
    user_brand_pref = {uid: np.random.choice(BRANDS) for uid in range(1, n_users + 1)}
    
    genders_col = np.random.choice(GENDERS, size=n_samples, p=[0.45, 0.45, 0.10])
    categories_col = np.random.choice(CATEGORIES, size=n_samples)
    brands_col = np.random.choice(BRANDS, size=n_samples)
    ptypes_col = np.random.choice(PRODUCT_TYPES, size=n_samples)
    freq_col = np.random.randint(1, 21, size=n_samples)

    records = []
    for i in range(n_samples):
        g, c, pt, uid = genders_col[i], categories_col[i], ptypes_col[i], user_ids[i]
        freq_factor = np.log(freq_col[i] + 1) / np.log(22) * 0.2
        prob = gender_affinity[g][c] + product_affinity[pt] * 0.2 + freq_factor + user_bias_map[uid]
        if brands_col[i] == user_brand_pref[uid]: prob += 0.10
        
        prob = min(max(prob, 0.05), 0.90)
        purchased = int(np.random.rand() < prob)
        if np.random.rand() < 0.25: purchased = 1 - purchased

        records.append({
            "user_id": uid, "category": c, "product_type": pt, "brand": brands_col[i],
            "gender": g, "purchase_frequency": freq_col[i], "purchased": purchased,
            "last_category": np.random.choice(CATEGORIES),
            "last_product_type": np.random.choice(PRODUCT_TYPES),
            "days_since_last_purchase": np.random.randint(0, 31)
        })
    return pd.DataFrame(records)

def build_user_memory(df: pd.DataFrame) -> pd.DataFrame:
    bought = df[df["purchased"] == 1]
    fav_cat = bought.groupby("user_id")["category"].agg(lambda x: x.value_counts().index[0])
    fav_brand = bought.groupby("user_id")["brand"].agg(lambda x: x.value_counts().index[0])
    fav_prod = bought.groupby("user_id")["product_type"].agg(lambda x: x.value_counts().index[0])
    avg_freq = df.groupby("user_id")["purchase_frequency"].mean().round(2)
    buy_rate = df.groupby("user_id")["purchased"].mean().round(3)

    memory = pd.concat([fav_cat, fav_brand, fav_prod, avg_freq, buy_rate], axis=1)
    memory.columns = ["fav_category", "fav_brand", "fav_product_type", "avg_frequency", "buy_rate"]
    return memory.reset_index()

def enrich_with_memory(df: pd.DataFrame, memory: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(memory, on="user_id", how="left")
    df["fav_category"] = df["fav_category"].fillna(COLD_START_DEFAULTS["fav_category"])
    df["fav_brand"] = df["fav_brand"].fillna(COLD_START_DEFAULTS["fav_brand"])
    df["fav_product_type"] = df["fav_product_type"].fillna(COLD_START_DEFAULTS["fav_product_type"])
    df["avg_frequency"] = df["avg_frequency"].fillna(COLD_START_DEFAULTS["avg_frequency"])
    df["buy_rate"] = df["buy_rate"].fillna(df["buy_rate"].median())
    
    df["is_fav_category"] = (df["category"] == df["fav_category"]).astype(int)
    df["is_fav_brand"] = (df["brand"] == df["fav_brand"]).astype(int)
    return df

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("num", "passthrough", NUMERIC_FEATURES),
    ])

def train_model(df: pd.DataFrame, model_type: str = "random_forest") -> Pipeline:
    X, y = df[FEATURES], df["purchased"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)
    else:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    pipeline = Pipeline([("preprocessor", build_preprocessor()), ("classifier", clf)])
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    print(f"[Training] Model: {model_type} | Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    return pipeline

def build_popularity_cache(df: pd.DataFrame) -> dict:
    pop_gender = df.groupby(["gender", "category", "product_type"])["purchased"].mean().to_dict()
    pop_global = df.groupby(["category", "product_type"])["purchased"].mean().to_dict()
    
    cache = {}
    for k, v in pop_gender.items(): cache[k] = round(v, 4)
    for k, v in pop_global.items(): cache[k] = round(v, 4)
    return cache

def save_artifacts(pipeline, memory, pop_cache):
    os.makedirs("app/ml_models", exist_ok=True)
    with open("app/ml_models/preference_model.pkl", "wb") as f: pickle.dump(pipeline, f)
    with open("app/ml_models/user_memory.pkl", "wb") as f: pickle.dump(memory, f)
    with open("app/ml_models/popularity_cache.pkl", "wb") as f: pickle.dump(pop_cache, f)
    print("[Success] Artifacts saved to app/ml_models/")

if __name__ == "__main__":
    print("--- Behavioral Preference Training Pipeline ---")
    raw_data = generate_dataset(n_samples=5000)
    user_mem = build_user_memory(raw_data)
    final_df = enrich_with_memory(raw_data, user_mem)
    
    trained_pipeline = train_model(final_df)
    popularity = build_popularity_cache(raw_data)
    
    save_artifacts(trained_pipeline, user_mem, popularity)
    print("--- Pipeline Complete ---")