import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


CATEGORIES     = ["streetwear", "casual", "formal", "sportswear", "oversized", "vintage"]
BRANDS         = ["Nike", "Zara", "H&M", "Adidas", "Uniqlo", "Gucci", "Pull&Bear"]
GENDERS        = ["M", "F", "Other"]
PRODUCT_TYPES  = ["t-shirt", "hoodie", "jacket", "jeans", "shorts", "dress", "sneakers"]

CATEGORICAL_FEATURES = [
    "category", "product_type", "brand", "gender",
    "fav_category", "fav_brand", "fav_product_type",
    "last_category", "last_product_type",
]
NUMERIC_FEATURES = [
    "purchase_frequency",
    "avg_frequency",
    "buy_rate",
    "days_since_last_purchase",
    "is_fav_category",
    "is_fav_brand",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

FIT_WEIGHTS = {
    "jeans":   0.60,
    "jacket":  0.55,
    "dress":   0.55,
    "t-shirt": 0.45,
    "hoodie":  0.45,
    "shorts":  0.45,
    "sneakers":0.40,
}

COLD_START_DEFAULTS = {
    "fav_category":             "casual",
    "fav_brand":                "unknown",
    "fav_product_type":         "t-shirt",
    "avg_frequency":            5,
    "buy_rate":                 0.5,
    "last_category":            "casual",
    "last_product_type":        "t-shirt",
    "days_since_last_purchase": 30,
    "purchase_frequency":       1,
    "is_fav_category":          0,
    "is_fav_brand":             0,
}


def generate_dataset(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    gender_affinity = {
        "M":     {"streetwear":0.80,"casual":0.60,"formal":0.30,
                  "sportswear":0.75,"oversized":0.70,"vintage":0.45},
        "F":     {"streetwear":0.50,"casual":0.75,"formal":0.60,
                  "sportswear":0.50,"oversized":0.55,"vintage":0.60},
        "Other": {"streetwear":0.60,"casual":0.65,"formal":0.40,
                  "sportswear":0.60,"oversized":0.65,"vintage":0.55},
    }
    product_affinity = {
        "t-shirt":0.70,"hoodie":0.80,"jacket":0.65,
        "jeans":  0.60,"shorts":0.55,"dress": 0.50,"sneakers":0.75,
    }

    n_users  = 300
    user_ids = np.random.randint(1, n_users + 1, size=n_samples)

    user_bias_map   = {uid: np.random.normal(0, 0.05) for uid in range(1, n_users + 1)}
    user_brand_pref = {uid: np.random.choice(BRANDS)  for uid in range(1, n_users + 1)}
    user_last_cat   = {uid: np.random.choice(CATEGORIES)     for uid in range(1, n_users + 1)}
    user_last_prod  = {uid: np.random.choice(PRODUCT_TYPES)  for uid in range(1, n_users + 1)}
    user_last_days  = {uid: np.random.randint(0, 31)         for uid in range(1, n_users + 1)}

    genders_col    = np.random.choice(GENDERS,        size=n_samples, p=[0.45, 0.45, 0.10])
    categories_col = np.random.choice(CATEGORIES,     size=n_samples)
    brands_col     = np.random.choice(BRANDS,         size=n_samples)
    ptypes_col     = np.random.choice(PRODUCT_TYPES,  size=n_samples)
    freq_col       = np.random.randint(1, 21,          size=n_samples)

    records = []
    for i in range(n_samples):
        g, c, pt, uid = genders_col[i], categories_col[i], ptypes_col[i], user_ids[i]

        freq_factor = np.log(freq_col[i] + 1) / np.log(22) * 0.2
        prob        = (gender_affinity[g][c]
                       + product_affinity[pt] * 0.2
                       + freq_factor
                       + user_bias_map[uid])

        if brands_col[i] == user_brand_pref[uid]:
            prob += 0.10

        prob      = min(max(prob, 0.05), 0.90)
        purchased = int(np.random.rand() < prob)
        if np.random.rand() < 0.30:
            purchased = 1 - purchased

        records.append({
            "user_id":                 uid,
            "category":                c,
            "product_type":            pt,
            "brand":                   brands_col[i],
            "gender":                  g,
            "purchase_frequency":      freq_col[i],
            "purchased":               purchased,
            "last_category":           user_last_cat[uid],
            "last_product_type":       user_last_prod[uid],
            "days_since_last_purchase":user_last_days[uid],
        })

    df = pd.DataFrame(records)
    print(f"[dataset]  {df.shape[0]} rows | "
          f"purchase rate: {df['purchased'].mean():.2%} | "
          f"users: {df['user_id'].nunique()}")
    return df


def build_user_memory(df: pd.DataFrame) -> pd.DataFrame:
    bought    = df[df["purchased"] == 1]
    fav_cat   = bought.groupby("user_id")["category"].agg(
                    lambda x: x.value_counts().index[0])
    fav_brand = bought.groupby("user_id")["brand"].agg(
                    lambda x: x.value_counts().index[0])
    fav_prod  = bought.groupby("user_id")["product_type"].agg(
                    lambda x: x.value_counts().index[0])
    avg_freq  = df.groupby("user_id")["purchase_frequency"].mean().round(2)
    buy_rate  = df.groupby("user_id")["purchased"].mean().round(3)

    memory = pd.concat([fav_cat, fav_brand, fav_prod, avg_freq, buy_rate], axis=1)
    memory.columns = ["fav_category", "fav_brand", "fav_product_type",
                      "avg_frequency", "buy_rate"]
    memory = memory.reset_index()
    print(f"[memory]   {len(memory)} user profiles built")
    return memory


def enrich_with_memory(df: pd.DataFrame, memory: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(
        memory[["user_id", "fav_category", "fav_brand",
                "fav_product_type", "avg_frequency", "buy_rate"]],
        on="user_id", how="left"
    )
    df["fav_category"]     = df["fav_category"].fillna(COLD_START_DEFAULTS["fav_category"])
    df["fav_brand"]        = df["fav_brand"].fillna(COLD_START_DEFAULTS["fav_brand"])
    df["fav_product_type"] = df["fav_product_type"].fillna(COLD_START_DEFAULTS["fav_product_type"])
    df["avg_frequency"]    = df["avg_frequency"].fillna(COLD_START_DEFAULTS["avg_frequency"])
    df["buy_rate"]         = df["buy_rate"].fillna(df["buy_rate"].median())

    df["is_fav_category"] = (df["category"] == df["fav_category"]).astype(int)
    df["is_fav_brand"]    = (df["brand"]    == df["fav_brand"]).astype(int)

    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
        ("num", "passthrough", NUMERIC_FEATURES),
    ])


def train_and_evaluate(df: pd.DataFrame,
                       model_choice: str = "random_forest") -> Pipeline:
    X, y = df[FEATURES], df["purchased"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    if model_choice == "logistic":
        clf = LogisticRegression(
            max_iter=1000, C=1.0,
            class_weight="balanced", random_state=42)

    elif model_choice == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=8,
            class_weight="balanced",
            random_state=42, n_jobs=-1)

    elif model_choice == "xgboost":
        from xgboost import XGBClassifier
        ratio = (y_train == 0).sum() / (y_train == 1).sum()
        clf = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=ratio,
            eval_metric="logloss", random_state=42)
    else:
        raise ValueError(f"Unknown model_choice: {model_choice!r}. "
                         f"Options: random_forest | logistic | xgboost")

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", clf),
    ])

    print(f"\n[train]    model={model_choice.upper()}  "
          f"train={len(X_train)}  test={len(X_test)}")
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    print(f"[eval]     accuracy={accuracy_score(y_test, y_pred):.4f}  "
          f"roc-auc={roc_auc_score(y_test, y_proba):.4f}")
    print(classification_report(y_test, y_pred,
                                 target_names=["won't buy", "will buy"]))
    return pipeline


def save_artifacts(pipeline: Pipeline,
                   memory: pd.DataFrame,
                   popularity_cache: dict,
                   model_path: str = "preference_model.pkl",
                   memory_path: str = "user_memory.pkl",
                   popularity_path: str = "popularity_cache.pkl") -> None:
    with open(model_path,      "wb") as f: pickle.dump(pipeline,         f)
    with open(memory_path,     "wb") as f: pickle.dump(memory,            f)
    with open(popularity_path, "wb") as f: pickle.dump(popularity_cache,  f)
    print(f"[saved]    {model_path}  |  {memory_path}  |  {popularity_path}")


def load_artifacts(model_path:      str = "preference_model.pkl",
                   memory_path:     str = "user_memory.pkl",
                   popularity_path: str = "popularity_cache.pkl"):
    with open(model_path,      "rb") as f: pipeline         = pickle.load(f)
    with open(memory_path,     "rb") as f: memory           = pickle.load(f)
    with open(popularity_path, "rb") as f: popularity_cache = pickle.load(f)
    return pipeline, memory, popularity_cache


def build_user_context(user_id: int,
                       memory: pd.DataFrame,
                       gender: str,
                       purchase_frequency: int,
                       last_category: str,
                       last_product_type: str,
                       days_since_last_purchase: int) -> dict:
    row = memory[memory["user_id"] == user_id]
    has_history = len(row) > 0

    if has_history:
        r = row.iloc[0]
        return {
            "gender":                  gender,
            "purchase_frequency":      purchase_frequency,
            "avg_frequency":           float(r["avg_frequency"]),
            "fav_category":            r["fav_category"],
            "fav_brand":               r["fav_brand"],
            "fav_product_type":        r["fav_product_type"],
            "buy_rate":                r["buy_rate"],
            "last_category":           last_category,
            "last_product_type":       last_product_type,
            "days_since_last_purchase":days_since_last_purchase,
            "_cold_start":             False,
        }
    else:
        return {
            "gender":                  gender,
            "purchase_frequency":      purchase_frequency,
            **{k: v for k, v in COLD_START_DEFAULTS.items()},
            "_cold_start":             True,
        }


def predict_preference(pipeline: Pipeline, user_ctx: dict, product: dict) -> dict:
    feat = {
        "category":                product["category"],
        "product_type":            product["product_type"],
        "brand":                   product["brand"],
        "gender":                  user_ctx["gender"],
        "purchase_frequency":      user_ctx["purchase_frequency"],
        "avg_frequency":           user_ctx.get("avg_frequency",
                                       COLD_START_DEFAULTS["avg_frequency"]),
        "fav_category":            user_ctx["fav_category"],
        "fav_brand":               user_ctx["fav_brand"],
        "fav_product_type":        user_ctx["fav_product_type"],
        "buy_rate":                user_ctx["buy_rate"],
        "last_category":           user_ctx["last_category"],
        "last_product_type":       user_ctx["last_product_type"],
        "days_since_last_purchase":user_ctx["days_since_last_purchase"],
        "is_fav_category": int(product["category"] == user_ctx["fav_category"]),
        "is_fav_brand":    int(product["brand"]    == user_ctx["fav_brand"]),
    }
    prob = pipeline.predict_proba(pd.DataFrame([feat]))[0][1]
    return {
        "preference_score": round(float(prob), 4),
        "will_buy":         prob >= 0.5,
        "label":            "LIKED" if prob >= 0.5 else "DISLIKED",
    }


def build_popularity_cache(df: pd.DataFrame) -> dict:
    pop_gender = (
        df.groupby(["gender", "category", "product_type"])["purchased"]
          .mean()
          .reset_index()
          .rename(columns={"purchased": "popularity"})
    )
    pop_global = (
        df.groupby(["category", "product_type"])["purchased"]
          .mean()
          .reset_index()
          .rename(columns={"purchased": "popularity"})
    )
    cache = {}
    for _, row in pop_gender.iterrows():
        cache[(row["gender"], row["category"], row["product_type"])] = round(row["popularity"], 4)
    for _, row in pop_global.iterrows():
        cache[(row["category"], row["product_type"])] = round(row["popularity"], 4)

    print(f"[pop cache] {len(cache)} entries  "
          f"({len(pop_gender)} gender-aware + {len(pop_global)} global fallbacks)")
    return cache


def get_popularity_score(popularity_cache: dict,
                         category: str,
                         product_type: str,
                         gender: str = "") -> float:
    if gender:
        score = popularity_cache.get((gender, category, product_type))
        if score is not None:
            return score
    return popularity_cache.get((category, product_type), 0.50)


def preference_mismatch_penalty(user_ctx: dict, product: dict) -> float:
    penalty = 0.0
    if product.get("category", "") != user_ctx.get("fav_category", ""):
        penalty += 0.05
    if product.get("product_type", "") != user_ctx.get("fav_product_type", ""):
        penalty += 0.03
    return round(penalty, 4)


def compute_final_score(fit_score:        float,
                        preference_score: float,
                        similarity_score: float,
                        nlp_score:        float = 0.0,
                        return_risk:      float = 0.0,
                        product_type:     str   = "",
                        user_ctx:         dict  = None,
                        product:          dict  = None) -> dict:
    w_fit  = FIT_WEIGHTS.get(product_type, 0.50)
    rem    = 1.0 - w_fit
    w_pref = round(rem * 0.45, 4)
    w_sim  = round(rem * 0.25, 4)
    w_nlp  = round(rem * 0.30, 4)
    w_risk = 0.15

    mismatch = (preference_mismatch_penalty(user_ctx, product)
                if user_ctx and product else 0.0)

    score = (w_fit  * fit_score
             + w_pref * preference_score
             + w_sim  * similarity_score
             + w_nlp  * nlp_score
             - w_risk * return_risk
             - mismatch)

    return {
        "final_score": round(score, 4),
        "mismatch_penalty": mismatch,
        "weights": {
            "fit":         w_fit,
            "preference":  w_pref,
            "similarity":  w_sim,
            "nlp":         w_nlp,
            "risk_penalty":w_risk,
        },
    }


def get_feature_names(pipeline: Pipeline) -> list:
    pre       = pipeline.named_steps["preprocessor"]
    ohe_names = pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    return list(ohe_names) + NUMERIC_FEATURES


def print_feature_importance(pipeline: Pipeline, top_n: int = 15) -> list:
    clf = pipeline.named_steps["classifier"]
    if not hasattr(clf, "feature_importances_"):
        print("[explain]  Feature importance requires a tree-based model.")
        return []

    names = get_feature_names(pipeline)
    pairs = sorted(zip(names, clf.feature_importances_),
                   key=lambda x: x[1], reverse=True)[:top_n]

    print(f"\n[explain]  Top {top_n} features:")
    for name, imp in pairs:
        print(f"  {name:<48} {imp:.4f}")
    return [(n, float(v)) for n, v in pairs]


def explain_recommendation(pipeline: Pipeline,
                            user_ctx: dict,
                            product: dict,
                            top_n: int = 5) -> dict:
    clf = pipeline.named_steps["classifier"]
    if not hasattr(clf, "feature_importances_"):
        return {"note": "explainability requires a tree-based model"}

    names   = get_feature_names(pipeline)
    pairs   = sorted(zip(names, clf.feature_importances_),
                     key=lambda x: x[1], reverse=True)[:top_n]
    pref    = predict_preference(pipeline, user_ctx, product)
    return {
        "preference_score": pref["preference_score"],
        "label":            pref["label"],
        "top_reasons":      [{"feature": n, "importance": round(float(v), 4)}
                             for n, v in pairs],
    }


def demo_returning_user(pipeline: Pipeline, memory: pd.DataFrame) -> None:
    print("\n" + "="*70)
    print("DEMO - returning user  |  Male  |  fav=streetwear/Nike/hoodie")
    print("="*70)

    user_ctx = build_user_context(
        user_id=42, memory=memory,
        gender="M", purchase_frequency=15,
        last_category="sportswear", last_product_type="sneakers",
        days_since_last_purchase=3,
    )
    print(f"  cold_start: {user_ctx['_cold_start']}  "
          f"fav={user_ctx['fav_category']}/{user_ctx['fav_product_type']}  "
          f"buy_rate={user_ctx['buy_rate']}")

    products = [
        {"id":"p1","category":"streetwear","product_type":"hoodie",  "brand":"Nike",
         "fit_score":0.92,"similarity_score":0.80},
        {"id":"p2","category":"streetwear","product_type":"jeans",   "brand":"Nike",
         "fit_score":0.88,"similarity_score":0.75},
        {"id":"p3","category":"casual",    "product_type":"t-shirt", "brand":"Uniqlo",
         "fit_score":0.90,"similarity_score":0.70},
        {"id":"p4","category":"formal",    "product_type":"jacket",  "brand":"Zara",
         "fit_score":0.85,"similarity_score":0.50},
        {"id":"p5","category":"sportswear","product_type":"sneakers","brand":"Adidas",
         "fit_score":0.78,"similarity_score":0.72},
        {"id":"p6","category":"formal",    "product_type":"dress",   "brand":"Gucci",
         "fit_score":0.60,"similarity_score":0.40},
    ]

    rows = []
    for prod in products:
        pref  = predict_preference(pipeline, user_ctx, prod)
        final = compute_final_score(
            fit_score        = prod["fit_score"],
            preference_score = pref["preference_score"],
            similarity_score = prod["similarity_score"],
            product_type     = prod["product_type"],
            user_ctx         = user_ctx,
            product          = prod,
        )
        rows.append({
            "ID":      prod["id"],
            "Cat":     prod["category"],
            "Type":    prod["product_type"],
            "Fit":     prod["fit_score"],
            "Pref":    pref["preference_score"],
            "Penalty": final["mismatch_penalty"],
            "FINAL":   final["final_score"],
            "Label":   pref["label"],
        })

    df_r = pd.DataFrame(rows).sort_values("FINAL", ascending=False)
    print(df_r.to_string(index=False))
    print(f"\n  #1  : {df_r.iloc[0]['Cat']}/{df_r.iloc[0]['Type']}  "
          f"final={df_r.iloc[0]['FINAL']}")
    print(f"  Last: {df_r.iloc[-1]['Cat']}/{df_r.iloc[-1]['Type']}  "
          f"final={df_r.iloc[-1]['FINAL']}")


def demo_cold_start(popularity_cache: dict) -> None:
    print("\n" + "="*70)
    print("DEMO - cold start  (new user, no purchase history)")
    print("="*70)
    gender   = "F"
    products = [
        {"id":"p1","category":"casual",    "product_type":"t-shirt",  "fit_score":0.91,"similarity_score":0.78},
        {"id":"p2","category":"streetwear","product_type":"hoodie",   "fit_score":0.80,"similarity_score":0.65},
        {"id":"p3","category":"formal",    "product_type":"dress",    "fit_score":0.87,"similarity_score":0.82},
        {"id":"p4","category":"sportswear","product_type":"sneakers", "fit_score":0.75,"similarity_score":0.60},
    ]
    rows = []
    for prod in products:
        pop   = get_popularity_score(popularity_cache,
                                     prod["category"], prod["product_type"],
                                     gender=gender)
        final = compute_final_score(prod["fit_score"], pop,
                                    prod["similarity_score"],
                                    product_type=prod["product_type"])
        rows.append({
            "ID":         prod["id"],
            "Cat":        prod["category"],
            "Type":       prod["product_type"],
            "Pop(F)":     pop,
            "FINAL":      final["final_score"],
        })
    df_r = pd.DataFrame(rows).sort_values("FINAL", ascending=False)
    print(df_r.to_string(index=False))


def demo_explainability(pipeline: Pipeline, memory: pd.DataFrame) -> None:
    print("\n" + "="*70)
    print("DEMO - explainability  (why was this recommended?)")
    print("="*70)
    user_ctx = build_user_context(
        user_id=42, memory=memory,
        gender="M", purchase_frequency=15,
        last_category="sportswear", last_product_type="sneakers",
        days_since_last_purchase=3,
    )
    product = {"category":"streetwear","product_type":"hoodie","brand":"Nike"}
    exp = explain_recommendation(pipeline, user_ctx, product)
    print(f"  score={exp['preference_score']}  [{exp['label']}]")
    print("  top reasons:")
    for r in exp["top_reasons"]:
        print(f"    {r['feature']:<48} {r['importance']:.4f}")


def demo_dynamic_weights() -> None:
    print("\n" + "="*70)
    print("DEMO - dynamic weights by product type")
    print("="*70)
    for pt in ["jeans", "jacket", "dress", "hoodie", "sneakers"]:
        w = compute_final_score(0.9, 0.8, 0.7, product_type=pt)["weights"]
        print(f"  {pt:<10} fit={w['fit']}  "
              f"pref={w['preference']}  "
              f"sim={w['similarity']}  "
              f"nlp={w['nlp']}  "
              f"risk_pen={w['risk_penalty']}")


if __name__ == "__main__":
    print("="*70)
    print("  MODULE 2 - BEHAVIORAL PREFERENCE MODELING")
    print("="*70)

    df = generate_dataset(n_samples=3000)
    memory = build_user_memory(df)
    df = enrich_with_memory(df, memory)
    pipeline = train_and_evaluate(df, model_choice="random_forest")
    popularity_cache = build_popularity_cache(df)

    save_artifacts(
        pipeline,
        memory,
        popularity_cache,
        model_path="app/ml_models/preference_model.pkl",
        memory_path="app/ml_models/user_memory.pkl",
        popularity_path="app/ml_models/popularity_cache.pkl"
    )

    demo_returning_user(pipeline, memory)
    demo_cold_start(popularity_cache)
    demo_explainability(pipeline, memory)
    demo_dynamic_weights()
    print_feature_importance(pipeline, top_n=15)