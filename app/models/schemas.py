from pydantic import BaseModel
from typing import List, Dict, Optional, Any

class UserFeatures(BaseModel):
    total_encuestas: int
    total_respuestas: int
    diversidad_respuestas: int
    tipos_respuesta: Dict[str, int]
    canales: Dict[str, int]
    temas_encuesta: Dict[str, int]
    patrones_respuesta: Dict[str, int]

class ClusterPredictionRequest(BaseModel):
    user_features: UserFeatures

class ModelTrainingRequest(BaseModel):
    n_clusters: Optional[int] = None
    max_k: Optional[int] = 6
