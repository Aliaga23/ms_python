from fastapi import APIRouter, HTTPException
from app.models import ModelTrainingRequest, ClusterPredictionRequest
from app.services.clustering_service import SurveyClusteringService

router = APIRouter()
clustering_service = SurveyClusteringService()

@router.post("/train-model")
async def train_model(request: ModelTrainingRequest):
    try:
        result = clustering_service.train_model(
            n_clusters=request.n_clusters,
            max_k=request.max_k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@router.post("/predict-cluster")
async def predict_cluster(request: ClusterPredictionRequest):
    try:
        result = clustering_service.predict_cluster(request.user_features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting cluster: {str(e)}")

@router.delete("/reset-model")
async def reset_model():
    try:
        clustering_service.reset_model()
        return {"message": "Model and data reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting model: {str(e)}")
