import sys
import os

# Ensure the root path is accessible to Python
sys.path.append(os.getcwd())

from app.services.ranking_service import RankingService, ProductScores

def test_ranking_service():
    print(f"{'='*60}")
    print("  RANKING SERVICE UNIT TEST")
    print(f"{'='*60}\n")
    
    ranking_service = RankingService()
    
    print(f"Loaded Weights: {ranking_service.weights}\n")
    
    # 1. Test calculation engine individually
    perfect_product = ProductScores(
        product_id="ideal_item",
        fit_score=1.0,           
        similarity_score=1.0,    
        preference_score=1.0,    
        nlp_score=1.0,           
        return_penalty=0.0       
    )
    
    bad_product = ProductScores(
        product_id="bad_item",
        fit_score=0.2,           
        similarity_score=0.1,    
        preference_score=0.3,    
        nlp_score=0.0,           
        return_penalty=0.4       
    )
    
    perf_score = ranking_service.calculate_final_score(perfect_product)
    bad_score = ranking_service.calculate_final_score(bad_product)
    
    print("--- Calculate Final Score Test ---")
    print(f"Perfect Product Expected ~1.0: Got {perf_score}")
    print(f"Bad Product Expected <0.2:   Got {bad_score}")
    
    # 2. Test explanation generator
    print("\n--- Explanation Generator Test ---")
    print(f"Perfect Product Explanation: {ranking_service.generate_explanation(perfect_product)}")
    print(f"Bad Product Explanation    : {ranking_service.generate_explanation(bad_product)}")
    
    # 3. Test Batch Orchestration (Ranking logic)
    candidate_data = [
        {
            "product_id": "item1_average",
            "fit_score": 0.6,
            "similarity_score": 0.5,
            "preference_score": 0.6,
            "nlp_score": 0.5,
            "return_penalty": 0.05
        },
        {
            "product_id": "item2_best",
            "fit_score": 0.9,
            "similarity_score": 0.8,
            "preference_score": 0.9,
            "nlp_score": 0.9,
            "return_penalty": 0.01
        },
        {
            "product_id": "item3_worst",
            "fit_score": 0.3,
            "similarity_score": 0.2,
            "preference_score": 0.2,
            "nlp_score": 0.0,
            "return_penalty": 0.3
        }
    ]
    
    print("\n--- Batch Ranking Orchestration Test ---")
    ranked_batch = ranking_service.rank_batch(candidate_data)
    
    for idx, rec in enumerate(ranked_batch, 1):
        print(f"Rank [{idx}] | ID: {rec['product_id']} | Final Score: {rec['final_score']:.4f}")
        print(f"           Explanation: {rec['explanation']}")
    
    # Validation logic
    assert ranked_batch[0]["product_id"] == "item2_best", "Sorting failed: Best item should be first"
    assert ranked_batch[-1]["product_id"] == "item3_worst", "Sorting failed: Worst item should be last"
    
    print(f"\n{'='*60}")
    print("  RANKING SERVICE TEST PASSED SUCCESSFULLY")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    test_ranking_service()
