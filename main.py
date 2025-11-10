from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

app = FastAPI(
    title="Survey K-means Clustering API",
    description="API para análisis de clustering K-means en datos de encuestas",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routes import survey_routes, model_routes

app.include_router(survey_routes.router, tags=["surveys"])
app.include_router(model_routes.router, tags=["models"])

@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "Survey K-means Clustering API",
        "version": "1.0.0",
        "endpoints": {
            "upload_data": "/upload-survey-data",
            "train_model": "/train-model",
            "predict": "/predict-cluster",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Endpoint de health check"""
    model_path = "survey_clustering_model.pkl"
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_trained": os.path.exists(model_path)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
