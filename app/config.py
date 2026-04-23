FIT_TOLERANCE = 15

CHEST_WEIGHT = 0.4
WAIST_WEIGHT = 0.3
HIP_WEIGHT = 0.3 
SHOULDER_WEIGHT = 0.0
HEIGHT_WEIGHT = 0.0
WEIGHT_WEIGHT = 0.0

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

# --- NLP Return Risk Configuration ---
SEMANTIC_THRESHOLD = 0.40
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

BODY_THRESHOLDS = {
    "shoulder": 44,   # cm — broad shoulders threshold
    "chest":    100,  # cm — wide chest threshold
    "waist":    85,   # cm — larger waist threshold
    "hips":     100,  # cm — wider hips threshold
}

# --- NLP Taxonomy & Keyword Settings ---
FEATURE_TAXONOMY = {
    "fit_type": {
        "slim":     ["slim", "skinny", "fitted", "figure-hugging", "tailored cut", "tapered"],
        "regular":  ["regular", "classic", "standard"],
        "oversized":["oversized", "relaxed", "boxy", "loose", "baggy"],
        "athletic": ["athletic", "performance", "compression"],
    },
    "fabric": {
        "cotton":   ["cotton", "jersey", "oxford"],
        "denim":    ["denim", "jeans"],
        "polyester":["polyester", "mesh", "synthetic"],
        "wool":     ["wool", "merino"],
        "stretch":  ["stretch", "elastane", "spandex", "flex"],
        "linen":    ["linen"],
    },
    "style": {
        "streetwear":["streetwear", "urban", "skate", "hoodie", "kangaroo"],
        "formal":    ["formal", "blazer", "notched lapel", "structured", "tailored"],
        "casual":    ["casual", "everyday", "relaxed"],
        "athletic":  ["running", "training", "athletic", "sport", "performance", "wicking"],
        "vintage":   ["vintage", "retro", "distressed", "faded", "washed"],
    },
    "special_properties": {
        "stretch":           ["stretch", "elastane", "spandex", "flexible"],
        "moisture_wicking":  ["moisture-wicking", "sweat-wicking", "wicking", "quick-dry"],
        "lightweight":       ["lightweight", "light", "breathable", "airy"],
        "heavyweight":       ["heavyweight", "thick", "structured", "padded"],
        "wrinkle_resistant": ["wrinkle-resistant", "wrinkle resistant", "non-iron"],
    },
    "silhouette": {
        "tapered":   ["tapered", "narrowing", "slim leg"],
        "a_line":    ["a-line", "flared", "flowing"],
        "straight":  ["straight", "regular leg"],
        "oversized": ["oversized", "boxy", "relaxed"],
    },
}

COMPLAINT_TAXONOMY = {
    "shoulder": ["tight on shoulders", "shoulder seam", "narrow shoulders", "shoulder too", "shoulders are too", "shoulder fit", "not for broad shoulders", "athletic build"],
    "chest": ["pulls across chest", "tight chest", "chest too", "chest area", "couldn't button", "can't button", "across the chest"],
    "waist": ["tight waist", "waist too tight", "tight at the waist", "waist is", "waistband tight", "waistband too"],
    "hips": ["tight hips", "around the hips", "hips too", "couldn't get them on", "hip area"],
    "thighs": ["thighs too tight", "tight thighs", "around the thighs", "thigh area"],
    "length": ["too short", "too long", "sleeves too long", "sleeves too short", "arms too long", "arms too short", "length is"],
    "sizing": ["runs small", "runs large", "runs smaller", "runs bigger", "size up", "size down", "smaller than expected", "larger than expected", "sizing is inconsistent", "true to size", "the sizing", "not true to size"],
}

COMPLAINT_ANCHORS = {
    "shoulder": "the shoulders are too tight, narrow, or don't fit well",
    "chest": "the chest area is too tight or the shirt pulls across the chest",
    "waist": "the waist is too tight or the waistband is uncomfortable",
    "hips": "the hips are too tight or hard to fit into",
    "thighs": "the thighs are too tight or restrictive",
    "length": "the item is too short or too long, sleeves or body length issue",
    "sizing": "this item runs small or large, not true to size, size up or down",
}

# --- ML Training & Feature Engineering ---
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

# --- Return Risk Optimization Configuration ---
FIT_TYPES = ["slim", "regular", "oversized", "athletic"]
SIZES     = ["XS", "S", "M", "L", "XL", "XXL"]

SIZE_CHEST_MAP = {
    "XS": 82, "S": 88, "M": 96, "L": 104, "XL": 112, "XXL": 120
}

CATEGORY_RETURN_BASE = {
    "formal":     0.35,
    "streetwear": 0.15,
    "casual":     0.18,
    "athletic":   0.20,
    "vintage":    0.22,
    "oversized":  0.12,
    "regular":    0.16,
    "slim":       0.30,
}

RETURN_RISK_FEATURES = [
    "fit_score", "size_mismatch", "morpho_similarity_score",
    "preference_score", "fit_risk_score", "personalized_nlp_penalty",
    "historical_return_rate", "size_inconsistency",
    "user_return_tendency", "body_mismatch_score",
]

# --- Global Ranking Engine Weights ---
GLOBAL_RANKING_WEIGHTS = {
    "fit": 0.35,
    "preference": 0.25,
    "similarity": 0.20,
    "nlp": 0.20
}

# --- Similarity Service Configuration ---
SIMILARITY_FEATURE_WEIGHTS = {
    'height': 1.2,
    'weight': 1.0,
    'chest': 1.2,
    'waist': 1.0,
    'hip': 1.0,
    'shoulder': 0.8
}