import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.nlp.model_loader import get_shared_model
from app.config import SEMANTIC_THRESHOLD, BODY_THRESHOLDS, COMPLAINT_TAXONOMY, COMPLAINT_ANCHORS



POSITIVE_SIGNALS = [
    "true to size", "fits perfectly", "perfect fit", "fits well",
    "fits true", "great fit", "love the fit", "fits as expected",
]

def extract_complaints_rule_based(review_text: str) -> dict:
    text = review_text.lower()
    positive_fit = any(phrase in text for phrase in POSITIVE_SIGNALS)
    complaints = []
    raw_matches = {}

    for body_part, keywords in COMPLAINT_TAXONOMY.items():
        matches_found = []
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                if kw == "true to size" and positive_fit:
                    continue
                matches_found.append(kw)
        if matches_found:
            complaints.append(body_part)
            raw_matches[body_part] = matches_found

    complaints = list(dict.fromkeys(complaints))
    return {
        "complaints": complaints,
        "sizing_issue": "sizing" in complaints,
        "positive_fit": positive_fit,
        "complaint_count": len(complaints),
        "raw_matches": raw_matches,
    }

def build_complaint_tfidf_index(reviews: list) -> tuple:
    texts = [r["text"] for r in reviews]
    review_ids = [r["review_id"] for r in reviews]
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words="english", max_features=300, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix, review_ids

def top_complaint_keywords(vectorizer, tfidf_matrix, review_idx: int, top_n: int = 5) -> list:
    row = tfidf_matrix[review_idx].toarray().flatten()
    top_ix = row.argsort()[::-1][:top_n]
    terms = vectorizer.get_feature_names_out()
    return [(terms[i], round(float(row[i]), 4)) for i in top_ix if row[i] > 0]

def find_reviews_similar_to_complaint(vectorizer, tfidf_matrix, review_ids: list, complaint_query: str, top_n: int = 3) -> list:
    query_vec = vectorizer.transform([complaint_query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_ix = sims.argsort()[::-1][:top_n]
    return [(review_ids[i], round(float(sims[i]), 4)) for i in top_ix]



def get_review_embeddings(texts: list) -> np.ndarray:
    try:
        model = get_shared_model()
        if model is None:
            raise Exception("Model not available")
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings, model
    except Exception:
        vec = TfidfVectorizer(max_features=128, stop_words="english")
        mat = vec.fit_transform(texts).toarray()
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return (mat / norms).astype(np.float32), None

def semantic_complaint_detection(review_text: str, model=None, threshold: float = SEMANTIC_THRESHOLD) -> list:
    try:
        if model is None:
            model = get_shared_model()
            if model is None:
                raise Exception("Model not available")
        review_emb = model.encode([review_text])
        anchor_embs = model.encode(list(COMPLAINT_ANCHORS.values()))
    except Exception:
        vec = TfidfVectorizer(max_features=128, stop_words="english")
        all_texts = list(COMPLAINT_ANCHORS.values()) + [review_text]
        mat = vec.fit_transform(all_texts).toarray()
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        mat = mat / norms
        anchor_embs = mat[:-1]
        review_emb = mat[[-1]]

    sims = cosine_similarity(review_emb, anchor_embs).flatten()
    detected = []
    for i, (body_part, _) in enumerate(COMPLAINT_ANCHORS.items()):
        if sims[i] >= threshold:
            detected.append((body_part, round(float(sims[i]), 3)))
    return detected

def compute_fit_risk_score(product_reviews: list, model=None) -> dict:
    if not product_reviews:
        return {
            "fit_risk_score": 0.0,
            "complaint_breakdown": {},
            "positive_review_ratio": 1.0,
            "review_count": 0,
            "dominant_complaint": None,
        }

    complaint_counts = {}
    positive_count = 0

    for review in product_reviews:
        text = review["text"]
        rule_result = extract_complaints_rule_based(text)
        if rule_result["positive_fit"]:
            positive_count += 1

        rule_complaints = set(rule_result["complaints"])
        semantic_complaints = set(body_part for body_part, _ in semantic_complaint_detection(text, model=model))

        all_complaints = rule_complaints | semantic_complaints
        for part in all_complaints:
            complaint_counts[part] = complaint_counts.get(part, 0) + 1

    total = len(product_reviews)
    total_complaint_signals = sum(complaint_counts.values())
    raw_risk = total_complaint_signals / (total * len(COMPLAINT_ANCHORS))
    fit_risk_score = round(min(raw_risk * 2.5, 1.0), 3)

    dominant = max(complaint_counts, key=complaint_counts.get) if complaint_counts else None

    return {
        "fit_risk_score": fit_risk_score,
        "complaint_breakdown": complaint_counts,
        "positive_review_ratio": round(positive_count / total, 3),
        "review_count": total,
        "dominant_complaint": dominant,
    }

def compute_personalized_penalty(user_measurements: dict, product_risk: dict) -> float:
    breakdown = product_risk.get("complaint_breakdown", {})
    penalty = 0.0

    for body_part, threshold in BODY_THRESHOLDS.items():
        user_val = user_measurements.get(body_part, 0)
        complaint_count = breakdown.get(body_part, 0)
        review_count = max(product_risk.get("review_count", 1), 1)

        if user_val > threshold and complaint_count > 0:
            complaint_rate = complaint_count / review_count
            body_excess = min((user_val - threshold) / threshold, 1.0)
            penalty += complaint_rate * body_excess * 0.25

    return round(min(penalty, 0.5), 3)

def get_nlp_penalty_for_ranking(user_measurements: dict, product_id: str, all_reviews: list, model=None) -> dict:
    product_reviews = [r for r in all_reviews if r["product_id"] == product_id]
    risk = compute_fit_risk_score(product_reviews, model=model)
    penalty = compute_personalized_penalty(user_measurements, risk)
    return {
        "product_id": product_id,
        "fit_risk_score": risk["fit_risk_score"],
        "personalized_penalty": penalty,
        "dominant_complaint": risk["dominant_complaint"],
        "complaint_breakdown": risk["complaint_breakdown"],
    }