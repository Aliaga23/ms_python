from fastapi import APIRouter, HTTPException
from app.services.gpt_analysis_service import GPTAnalysisService
from typing import Dict

router = APIRouter()

@router.post("/analyze-with-gpt")
async def analyze_clustering_with_gpt(clustering_results: Dict):
    try:
        gpt_service = GPTAnalysisService()
        enhanced_results = gpt_service.analyze_clustering_results(clustering_results)
        
        if "error" in enhanced_results:
            raise HTTPException(status_code=500, detail=enhanced_results["error"])
        
        return enhanced_results
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis GPT: {str(e)}")
