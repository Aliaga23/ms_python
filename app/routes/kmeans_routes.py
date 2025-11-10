from fastapi import APIRouter, HTTPException
from app.services.kmeans_service import KMeansService
from app.services.gpt_analysis_service import GPTAnalysisService
from typing import Dict, List, Union

router = APIRouter()
kmeans_service = KMeansService()

@router.post("/analyze-survey-kmeans")
async def analyze_survey_kmeans(survey_data: Union[Dict, List[Dict]]):
    try:
        result = kmeans_service.analyze_survey(survey_data)
        
        kmeans_service.reset_model()
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        kmeans_service.reset_model()
        raise HTTPException(status_code=500, detail=f"Error analyzing survey: {str(e)}")

@router.post("/analyze-survey-complete")
async def analyze_survey_complete(survey_data: Union[Dict, List[Dict]]):
    try:
        clustering_result = kmeans_service.analyze_survey(survey_data)
        
        kmeans_service.reset_model()
        
        if "error" in clustering_result:
            raise HTTPException(status_code=400, detail=clustering_result["error"])
        
        try:
            gpt_service = GPTAnalysisService()
            enhanced_result = gpt_service.analyze_clustering_results(clustering_result)
            
            if "error" in enhanced_result:
                return {
                    "clustering": clustering_result,
                    "gpt_analysis": None,
                    "gpt_error": enhanced_result["error"],
                    "note": "Clustering exitoso pero análisis GPT falló"
                }
            
            return enhanced_result
            
        except ValueError as ve:
            return {
                "clustering": clustering_result,
                "gpt_analysis": None,
                "gpt_error": str(ve),
                "note": "Clustering exitoso pero GPT requiere OPENAI_API_KEY"
            }
        except Exception as gpt_error:
            return {
                "clustering": clustering_result,
                "gpt_analysis": None,
                "gpt_error": str(gpt_error),
                "note": "Clustering exitoso pero análisis GPT falló"
            }
        
    except Exception as e:
        kmeans_service.reset_model()
        raise HTTPException(status_code=500, detail=f"Error analyzing survey: {str(e)}")
