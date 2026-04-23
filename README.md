# 🧠 AI Recommendation Service

> Intelligent product recommendation engine for the **3D Fashion Marketplace** — built with FastAPI, Scikit-Learn, and NLP.

This microservice powers the AI layer of the marketplace. It receives user profiles and product catalogs, then returns a **ranked list of recommendations** by combining six specialized ML engines into a single unified score.

---

## ⚙️ Architecture Overview

```
POST /recommend
      │
      ▼
┌─────────────────────────────────────────────┐
│              Ranking Service                │
│         (Weighted Aggregation)              │
│                                             │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ │
│  │    NLP    │ │Preference │ │    Fit     │ │
│  │  Engine   │ │  Engine   │ │  Engine    │ │
│  └───────────┘ └───────────┘ └───────────┘ │
│  ┌───────────┐ ┌───────────┐               │
│  │ Similarity│ │  Return   │               │
│  │  Engine   │ │   Risk    │               │
│  └───────────┘ └───────────┘               │
└─────────────────────────────────────────────┘
      │
      ▼
  Ranked JSON Response
```

---

## 🔬 AI/ML Services

### 1. NLP Engine — Text Comprehension & Parsing
**`app/nlp/description_parser.py`** · **`app/nlp/review_analyzer.py`**

- **TF-IDF Vectorization** — Turns product descriptions into mathematical vectors, measuring word importance across the catalog.
- **Cosine Similarity** — Computes the semantic alignment between a user's search query and each product description.
- **Regex Taxonomy Rules** — Extracts structured features from raw text (e.g. "elastane" → `has_stretch: true`).
- **Review Sentiment Analysis** — Analyzes user reviews to detect fit complaints and applies personalized NLP penalties.

### 2. Preference Engine — User Affinity Prediction
**`app/services/preference_service.py`**

- **Random Forest Classifier** — Predicts buy probability from behavioral features: `days_since_last_purchase`, `price_sensitivity`, `favorite_brand`, etc.
- **Cold Start Handling** — Falls back to a popularity-based score cache when new users have no purchase history.

### 3. Fit Engine — Morphological Matching
**`app/services/fit_service.py`**

- **Non-Linear Gaussian Scoring** — Bell-curve scoring that forgives minor measurement differences but penalizes large ones exponentially.
- **Asymmetric Penalty Logic** — "Too tight" penalizes harder than "too loose" (tight clothes are unwearable; loose clothes can be styled).

### 4. Return Risk Engine — Loss Prevention
**`app/services/return_service.py`**

- **Probabilistic ML Classifier** — Evaluates return risk from features like `size_mismatch`, `historical_return_rate`, and NLP penalty.
- **Exponential Penalty Function** — `penalty = (prob²) × 0.5`. Low risk (<30%) = no penalty. High risk = aggressive ranking demotion.

### 5. Similarity Engine — Body Twin Matching
**`app/services/similarity.py`**

- **StandardScaler Normalization** — Ensures height (180cm) doesn't overshadow smaller-variance measurements like shoulder width (40cm).
- **K-Nearest Neighbors (KNN)** — Finds users with the closest physical measurements. If your body twins loved an item, you likely will too.

### 6. Ranking Service — Master Orchestrator
**`app/services/ranking_service.py`**

- **Weighted Aggregation** — `(Fit × 0.35) + (Preference × 0.25) + (Similarity × 0.20) + (NLP × 0.20) − Return_Penalty`
- **Threshold Explanations** — Generates human-readable labels (e.g. `"matches your style · highly relevant"`) for the front-end.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/fit-score` | Calculate fit score between user measurements and a garment |
| `POST` | `/similar-users` | Find morphologically similar users (body twins) |
| `POST` | `/recommend` | Full recommendation pipeline — returns ranked products |

---

## 🗂 Project Structure

```
ai-service/
├── app/
│   ├── main.py                    # FastAPI application & endpoints
│   ├── config.py                  # ML weights, thresholds, taxonomy
│   ├── models/
│   │   └── schema.py              # Pydantic request/response schemas
│   ├── nlp/
│   │   ├── description_parser.py  # TF-IDF & text feature extraction
│   │   ├── review_analyzer.py     # Review sentiment & NLP penalties
│   │   └── model_loader.py        # Shared embedding model singleton
│   ├── services/
│   │   ├── fit_service.py         # Gaussian fit scoring
│   │   ├── preference_service.py  # Random Forest preference prediction
│   │   ├── similarity.py          # KNN body twin matching
│   │   ├── return_service.py      # Return risk prediction & penalty
│   │   └── ranking_service.py     # Weighted aggregation & ranking
│   ├── trainings/
│   │   ├── behavior_preference.py # Preference model training script
│   │   └── return_risk_trainer.py # Return risk model training script
│   └── data/                      # Mock data for development
├── tests/
│   ├── fit_service.py             # Fit engine tests
│   ├── preference.py              # Preference engine tests
│   ├── similarity.py              # Similarity engine tests
│   ├── recommendation.py          # Full pipeline tests
│   ├── nlp_penalty.py             # NLP penalty tests
│   └── verify_ranking.py          # Ranking engine tests
└── scripts/                       # Utility scripts
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/FatmaMejri1/AI-Powered-Fashion-Marketplace.git
cd ai-service

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install fastapi==0.110.0 uvicorn[standard]==0.29.0 \
    sqlalchemy==2.0.29 psycopg2-binary==2.9.9 \
    numpy==1.26.4 pandas==2.2.1 scikit-learn==1.4.2 \
    python-multipart==0.0.9 pydantic==2.6.4 spacy==3.8.14
```

### Run the server

```bash
uvicorn app.main:app --reload --port 8002
```

The API will be available at `http://localhost:8002` with interactive docs at `/docs`.

---

## 🧪 Running Tests

```bash
# Individual service tests
python -m tests.fit_service
python -m tests.preference
python -m tests.similarity
python -m tests.nlp_penalty
python -m tests.verify_ranking

# Full recommendation pipeline
python -m tests.recommendation
```

---

## 🔧 Tech Stack

| Technology | Purpose |
|-----------|---------|
| **FastAPI** | REST API framework |
| **Scikit-Learn** | ML models (Random Forest, KNN, TF-IDF) |
| **spaCy** | NLP text processing |
| **Sentence-Transformers** | Semantic embeddings (all-MiniLM-L6-v2) |
| **NumPy / Pandas** | Numerical computation & data manipulation |
| **Pydantic** | Request/response validation |

---

## 🏗 Integration

This service is designed to be called by the **Node.js backend** of the 3D Fashion Marketplace. The backend forwards user context and product catalog data, and the AI service returns ranked recommendations with explainable scores.

```
Frontend (Angular) → Backend (Node.js) → AI Service (FastAPI) → Ranked Response
```
