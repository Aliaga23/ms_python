from fastapi import APIRouter, HTTPException
from app.models import TextAnalysisRequest
from app.services.text_analysis_service import TextAnalysisService
from typing import List, Dict

router = APIRouter()
text_service = TextAnalysisService()

@router.post("/analyze-text-complete")
async def analyze_text_complete(respuestas: List[Dict]):
    try:
        text_service.train_model(respuestas)
        
        result = text_service.analyze_responses(respuestas)
        
        text_service.reset_model()
        
        return result
    except Exception as e:
        text_service.reset_model()
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")

@router.post("/analyze-text")
async def analyze_text(respuestas: List[Dict]):
    try:
        result = text_service.analyze_responses(respuestas)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")

@router.post("/train-text-model")
async def train_text_model(respuestas: List[Dict]):
    try:
        result = text_service.train_model(respuestas)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@router.post("/predict-text")
async def predict_single_text(data: dict):
    try:
        texto = data.get('texto', '')
        if not texto:
            raise HTTPException(status_code=400, detail="Text field is required")
        
        result = text_service.predict_single(texto)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting: {str(e)}")

@router.delete("/reset-text-model")
async def reset_text_model():
    try:
        text_service.reset_model()
        return {"message": "Text analysis model reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting model: {str(e)}")

@router.get("/text-model-status")
async def get_text_model_status():
    return {
        "is_trained": text_service.is_trained,
        "model_exists": text_service.load_model() if not text_service.is_trained else True
    }
