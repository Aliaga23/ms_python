from fastapi import APIRouter, HTTPException, UploadFile, File
import json
from app.services.clustering_service import SurveyClusteringService

router = APIRouter()
clustering_service = SurveyClusteringService()

@router.post("/upload-survey-data")
async def upload_survey_data(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="File must be a JSON file")
        
        content = await file.read()
        json_data = json.loads(content.decode('utf-8'))
        
        success = clustering_service.load_survey_data(json_data)
        
        if success:
            if isinstance(json_data, list):
                total_entregas = sum(
                    len(set([r.get('entrega_id') for r in survey.get('respuestas_usuario', []) if r.get('entrega_id')])) 
                    for survey in json_data
                )
                return {
                    "message": "Survey data uploaded successfully",
                    "format": "survey_list",
                    "surveys_count": len(json_data),
                    "deliveries_count": total_entregas,
                    "total_responses": sum(len(s.get('respuestas_usuario', [])) for s in json_data)
                }
            else:
                return {
                    "message": "Survey data uploaded successfully",
                    "format": "user_dict",
                    "users_count": len(json_data.get('usuarios', [])),
                    "total_surveys": json_data.get('estadisticas', {}).get('total_encuestas', 0),
                    "total_responses": json_data.get('estadisticas', {}).get('total_respuestas', 0)
                }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading data: {str(e)}")

@router.post("/upload-survey-data-direct")
async def upload_survey_data_direct(survey_data: dict):
    try:
        success = clustering_service.load_survey_data(survey_data)
        
        if success:
            if isinstance(survey_data, list):
                total_entregas = sum(
                    len(set([r.get('entrega_id') for r in survey.get('respuestas_usuario', []) if r.get('entrega_id')])) 
                    for survey in survey_data
                )
                return {
                    "message": "Survey data uploaded successfully",
                    "format": "survey_list",
                    "surveys_count": len(survey_data),
                    "deliveries_count": total_entregas
                }
            else:
                return {
                    "message": "Survey data uploaded successfully",
                    "format": "user_dict",
                    "users_count": len(survey_data.get('usuarios', [])),
                    "total_surveys": survey_data.get('estadisticas', {}).get('total_encuestas', 0),
                    "total_responses": survey_data.get('estadisticas', {}).get('total_respuestas', 0)
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading data: {str(e)}")
