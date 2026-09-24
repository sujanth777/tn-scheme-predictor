"""
FastAPI Backend Application for TN Scheme Eligibility Predictor.
Exposes RESTful endpoints for /predict, /schemes, and /metrics.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List

try:
    from backend.schemas import CitizenProfileRequest, PredictionResponse
    from backend.predictor import WelfarePredictor
except ImportError:
    from schemas import CitizenProfileRequest, PredictionResponse
    from predictor import WelfarePredictor

# Initialize FastAPI application
app = FastAPI(
    title="TN Scheme Eligibility Predictor API",
    description="Production-grade REST API evaluating citizen eligibility across 20 Tamil Nadu welfare schemes using MultiOutput XGBoost & SHAP explainability.",
    version="2.0.0"
)

# Enable CORS for React frontend (Vite default :5173, Next.js default :3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate ML predictor engine
try:
    predictor = WelfarePredictor()
except Exception as e:
    predictor = None
    print(f"Warning: Failed to load WelfarePredictor at startup: {e}")


@app.get("/", tags=["General"])
def read_root():
    return {
        "service": "TN Scheme Eligibility Predictor API",
        "version": "2.0.0",
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "health": "/api/health",
            "schemes": "/api/schemes",
            "metrics": "/api/metrics",
            "predict": "/api/predict"
        }
    }


@app.get("/api/health", tags=["Health"])
def health_check():
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor engine is not initialized or model artifacts are missing."
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "total_schemes": len(predictor.rules_df),
        "feature_count": len(predictor.feature_columns)
    }


@app.get("/api/schemes", tags=["Schemes"])
def get_schemes():
    """Retrieve full catalog of 20 Tamil Nadu welfare schemes and their official rules."""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor service uninitialized.")
    try:
        schemes = predictor.get_all_schemes()
        return {
            "status": "success",
            "count": len(schemes),
            "schemes": schemes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch scheme directory: {str(e)}")


@app.get("/api/metrics", tags=["Metrics"])
def get_metrics():
    """Retrieve model performance benchmarks, multi-model comparisons, and per-scheme test F1 scores."""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor service uninitialized.")
    try:
        data = predictor.get_metrics()
        return {
            "status": "success",
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch metrics: {str(e)}")


@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_eligibility(profile: CitizenProfileRequest):
    """
    Accepts 19-attribute citizen profile, executes one-hot encoding parity,
    evaluates MultiOutputClassifier XGBoost model, computes SHAP explanations,
    and returns ranked eligibility decisions.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not ready to process predictions.")
    try:
        profile_dict = profile.model_dump()
        result = predictor.predict(profile_dict)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
