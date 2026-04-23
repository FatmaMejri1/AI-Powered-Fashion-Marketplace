import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.config import FEATURE_TAXONOMY



def extract_features_rule_based(description: str) -> dict:
    text = description.lower()
    features = {}

    for feature_name, categories in FEATURE_TAXONOMY.items():
        detected = []
        for category, keywords in categories.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', text):
                    detected.append(category)
                    break
        features[feature_name] = list(dict.fromkeys(detected)) if detected else ["unknown"]

    features["has_stretch"] = bool(re.search(r'\b(stretch|elastane|spandex|\d+%\s*elastane)\b', text))
    features["has_wicking"] = bool(re.search(r'\b(wicking|moisture.wicking|sweat.wicking)\b', text))
    features["has_lining"]  = bool(re.search(r'\b(lined|lining|inner|built.in liner)\b', text))

    care = []
    if re.search(r'machine washable', text): care.append("machine_wash")
    if re.search(r'dry clean',         text): care.append("dry_clean_only")
    if re.search(r'hand wash',         text): care.append("hand_wash")
    features["care"] = care if care else ["not_specified"]

    return features

def build_tfidf_index(catalog: list) -> tuple:
    descriptions = [p["description"] for p in catalog]
    ids          = [p["id"]          for p in catalog]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=500, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    return vectorizer, tfidf_matrix, ids

def top_keywords_for_product(vectorizer, tfidf_matrix, product_idx: int, top_n: int = 8) -> list:
    row    = tfidf_matrix[product_idx].toarray().flatten()
    top_ix = row.argsort()[::-1][:top_n]
    terms  = vectorizer.get_feature_names_out()
    return [(terms[i], round(float(row[i]), 4)) for i in top_ix if row[i] > 0]

def find_similar_by_text(vectorizer, tfidf_matrix, ids: list, query_description: str, top_n: int = 3) -> list:
    query_vec = vectorizer.transform([query_description])
    sims      = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_ix    = sims.argsort()[::-1][:top_n]
    return [(ids[i], round(float(sims[i]), 4)) for i in top_ix]

def get_sentence_embeddings(descriptions: list) -> np.ndarray:
    vec  = TfidfVectorizer(max_features=128, stop_words="english")
    mat  = vec.fit_transform(descriptions).toarray()
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (mat / norms).astype(np.float32)

def build_semantic_index(catalog: list) -> tuple:
    descriptions = [p["description"] for p in catalog]
    ids          = [p["id"]          for p in catalog]
    embeddings   = get_sentence_embeddings(descriptions)
    return embeddings, ids

def semantic_search(embeddings: np.ndarray, ids: list, query: str, top_n: int = 3) -> list:
    from app.data.mock_catalog import CATALOG
    vec       = TfidfVectorizer(max_features=128, stop_words="english")
    all_texts = [p["description"] for p in CATALOG] + [query]
    mat       = vec.fit_transform(all_texts).toarray()
    norms     = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    mat       = mat / norms
    query_emb = mat[[-1]]
    embeddings = mat[:-1]

    sims   = cosine_similarity(query_emb, embeddings).flatten()
    top_ix = sims.argsort()[::-1][:top_n]
    return [(ids[i], round(float(sims[i]), 4)) for i in top_ix]

def process_catalog(catalog: list) -> pd.DataFrame:
    rule_rows = []
    for p in catalog:
        feats = extract_features_rule_based(p["description"])
        row   = {"id": p["id"], "name": p["name"]}
        for k, v in feats.items():
            if isinstance(v, list):
                row[k] = ", ".join(v)
            else:
                row[k] = v
        rule_rows.append(row)
    df = pd.DataFrame(rule_rows)

    vectorizer, tfidf_matrix, ids = build_tfidf_index(catalog)
    keywords_col = []
    for i in range(len(catalog)):
        kw = top_keywords_for_product(vectorizer, tfidf_matrix, i, top_n=5)
        keywords_col.append(", ".join([f"{w}({s})" for w, s in kw]))
    df["top_keywords"] = keywords_col

    embeddings, emb_ids = build_semantic_index(catalog)
    return df, vectorizer, tfidf_matrix, embeddings, ids

def analyze_description(description: str) -> dict:
    features = extract_features_rule_based(description)
    return {
        "input":    description,
        "features": features,
    }

def enrich_product_for_recommender(product_id: str, description: str, catalog_df: pd.DataFrame) -> dict:
    feats = extract_features_rule_based(description)
    fit_map = {
        "slim": "slim", "oversized": "oversized",
        "athletic": "athletic", "regular": "regular",
    }
    fit_types = feats.get("fit_type", ["unknown"])
    fit_label = fit_map.get(fit_types[0], "regular") if fit_types else "regular"

    return {
        "product_id":   product_id,
        "fit_type":     fit_label,
        "fabric":       feats.get("fabric", ["unknown"])[0],
        "style":        feats.get("style",  ["unknown"])[0],
        "has_stretch":  feats.get("has_stretch", False),
        "has_wicking":  feats.get("has_wicking", False),
        "silhouette":   feats.get("silhouette", ["unknown"])[0],
        "raw_features": feats,
    }