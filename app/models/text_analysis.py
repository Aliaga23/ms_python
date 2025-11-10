from pydantic import BaseModel
from typing import List, Dict, Optional

class TextAnalysisRequest(BaseModel):
    respuestas: List[Dict[str, str]]

class AnomalyDetectionResult(BaseModel):
    respuesta_id: str
    texto_respuesta: str
    is_anomaly: bool
    anomaly_score: float
    pregunta_texto: Optional[str] = None

class ClassificationResult(BaseModel):
    respuesta_id: str
    texto_respuesta: str
    categoria_predicha: str
    confianza: float
    sentimiento: str  # 'positivo', 'negativo', 'neutro'
    sentimiento_score: float  # -1.0 (muy negativo) a 1.0 (muy positivo)
    pregunta_texto: Optional[str] = None

class TextAnalysisResponse(BaseModel):
    total_respuestas: int
    anomalias_detectadas: int
    anomalias: List[AnomalyDetectionResult]
    ejemplos_por_categoria: List[ClassificationResult]  # Solo una muestra por categoría
    categorias_encontradas: Dict[str, int]
    sentimiento_resumen: Dict[str, int]  # conteo de positivos, negativos, neutros
    sentimiento_por_categoria: Dict[str, Dict[str, int]]  # sentimiento desglosado por categoría
