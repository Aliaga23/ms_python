from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
import warnings
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

app = FastAPI(
    title="Survey Analysis API",
    description="API para análisis de clustering K-means y análisis de texto con ML",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routes import survey_routes, model_routes, text_analysis_routes, kmeans_routes, gpt_analysis_routes

app.include_router(survey_routes.router, tags=["surveys"])
app.include_router(model_routes.router, tags=["models"])
app.include_router(text_analysis_routes.router, tags=["text-analysis"])
app.include_router(kmeans_routes.router, tags=["kmeans"])
app.include_router(gpt_analysis_routes.router, tags=["gpt-analysis"])

@app.get("/")
async def root():
    return {
        "message": "Survey Analysis API",
        "version": "2.0.0",
        "endpoints": {
            "clustering": {
                "upload_data": "/upload-survey-data",
                "train_model": "/train-model",
                "predict": "/predict-cluster"
            },
            "text_analysis": {
                "analyze_complete": "/analyze-text-complete",
                "analyze": "/analyze-text",
                "train": "/train-text-model",
                "predict": "/predict-text",
                "status": "/text-model-status"
            },
            "kmeans": {
                "analyze_survey": "/analyze-survey-kmeans",
                "analyze_complete_with_gpt": "/analyze-survey-complete"
            },
            "gpt_analysis": {
                "analyze_clustering": "/analyze-with-gpt"
            },
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    model_path = "survey_clustering_model.pkl"
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_trained": os.path.exists(model_path)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
