<<<<<<< Updated upstream
#  AI Recommendation Service

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




## 📡 API Endpoints (AI/ML)

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

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
=======
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




## 📡 API Endpoints (AI/ML)

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

---

## ?? License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
>>>>>>> Stashed changes
