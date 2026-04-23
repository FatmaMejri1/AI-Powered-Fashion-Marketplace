import math
from dataclasses import dataclass, field
from typing import Optional, List
from app.config import GLOBAL_RANKING_WEIGHTS as W

@dataclass
class ProductScores:
    product_id:           str
    fit_score:            float = 0.0
    similarity_score:     float = 0.0
    preference_score:     float = 0.0
    nlp_score:            float = 0.0
    return_penalty:       float = 0.0
    final_score:          float = 0.0
    explanation:          str   = ""

class RankingService:
    def __init__(self):
        self.weights = W

    def calculate_final_score(self, scores: ProductScores) -> float:
        """
        Calculates the weighted score based on GLOBAL_RANKING_WEIGHTS.
        """
        raw_score = (
            self.weights["fit"]        * scores.fit_score +
            self.weights["preference"] * scores.preference_score +
            self.weights["similarity"] * scores.similarity_score +
            self.weights["nlp"]         * scores.nlp_score -
            scores.return_penalty
        )
        return round(max(0.0, min(1.0, raw_score)), 4)

    def generate_explanation(self, scores: ProductScores) -> str:
        parts = []
        if scores.fit_score >= 0.8: parts.append("excellent fit")
        if scores.preference_score >= 0.7: parts.append("matches your style")
        if scores.similarity_score >= 0.7: parts.append("popular with your body twins")
        if scores.nlp_score >= 0.6: parts.append("relevant to your search")
        if scores.return_penalty > 0.1: parts.append("[!] fit risk detected")
        
        return " · ".join(parts) if parts else "recommended"

    def rank_batch(self, candidate_data: List[dict]) -> List[dict]:
        """
        Orchestrates scoring and sorting for the final API response.
        """
        results = []
        for data in candidate_data:
            scores = ProductScores(
                product_id=data["product_id"],
                fit_score=data.get("fit_score", 0.0),
                similarity_score=data.get("similarity_score", 0.0),
                preference_score=data.get("preference_score", 0.0),
                nlp_score=data.get("nlp_score", 0.0),
                return_penalty=data.get("return_penalty", 0.0)
            )
            
            scores.final_score = self.calculate_final_score(scores)
            scores.explanation = self.generate_explanation(scores)
            
            # Map back to dict for API response
            data.update({
                "final_score": scores.final_score,
                "explanation": scores.explanation,
                "weights": self.weights
            })
            results.append(data)

        # Sort descending
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results
