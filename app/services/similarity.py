import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from app.config import SIMILARITY_FEATURE_WEIGHTS

logger = logging.getLogger(__name__)

class SimilarityService:
    def __init__(self, users_data):
        self.raw_data = users_data
        self.scaler = StandardScaler()
        self.knn = NearestNeighbors(metric='euclidean', algorithm='auto')
        self.feature_weights = SIMILARITY_FEATURE_WEIGHTS
        self.features = list(self.feature_weights.keys())

    def _extract_features(self, user):
        vec = []
        for feature in self.features:
            val = float(getattr(user, feature, 0) if hasattr(user, feature) else user.get(feature, 0))
            vec.append(val * self.feature_weights[feature])
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