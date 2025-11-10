from fastapi import APIRouter, HTTPException
from app.services.kmeans_service import KMeansService
from typing import Dict

router = APIRouter()
kmeans_service = KMeansService()

@router.post("/analyze-survey-kmeans")
async def analyze_survey_kmeans(survey_data: Dict):
    try:
        result = kmeans_service.analyze_survey(survey_data)
        
        kmeans_service.reset_model()
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        kmeans_service.reset_model()
        raise HTTPException(status_code=500, detail=f"Error analyzing survey: {str(e)}")
