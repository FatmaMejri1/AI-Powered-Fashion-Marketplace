import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class SimilarityService:
    FEATURE_WEIGHTS = {
        'height': 1.2,
        'weight': 1.0,
        'chest': 1.2,
        'waist': 1.0,
        'hip': 1.0,
        'shoulder': 0.8
    }
    FEATURES = list(FEATURE_WEIGHTS.keys())

    def __init__(self, users_data):
        self.raw_data = users_data
        self.scaler = StandardScaler()
        self.knn = NearestNeighbors(metric='euclidean', algorithm='auto')
        self.is_fitted = False

    def _extract_features(self, user):
        vec = []
        for feature in self.FEATURES:
            val = float(getattr(user, feature, 0) if hasattr(user, feature) else user.get(feature, 0))
            vec.append(val * self.FEATURE_WEIGHTS[feature])
        return vec

    def find_similar_users(self, target_user, k=5):
        target_gender = getattr(target_user, 'gender', 'unisex').lower()
        
        filtered_data = [
            u for u in self.raw_data 
            if str(u.get('gender', 'unisex')).lower() == target_gender
        ]

        if not filtered_data:
            logger.warning(f"No users found for gender: {target_gender}")
            return []

        try:
            measurements = [self._extract_features(u) for u in filtered_data]
            scaled_data = self.scaler.fit_transform(np.array(measurements))
            self.knn.fit(scaled_data)

            target_vec = np.array([self._extract_features(target_user)])
            target_scaled = self.scaler.transform(target_vec)

            n_neighbors = min(k, len(filtered_data))
            distances, indices = self.knn.kneighbors(target_scaled, n_neighbors=n_neighbors)

            results = []
            for i, idx in enumerate(indices[0]):
                user = filtered_data[idx]
                results.append({
                    "id": user.get('id'),
                    "distance": round(float(distances[0][i]), 3),
                    "similarity_score": round(max(0, 1 - distances[0][i]/10), 3)
                })
            
            return results
        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            return []