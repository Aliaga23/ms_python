from fastapi import HTTPException
from typing import Optional, List
import pickle
import os
import numpy as np
from app.models import UserFeatures
from app.utils import clean_for_json

class SurveyClusteringService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.analyzer = None
            cls._instance.model_path = "survey_clustering_model.pkl"
            cls._instance._model_cache = None
        return cls._instance
    
    def load_survey_data(self, json_data):
        """Carga datos de encuesta desde JSON - optimizado"""
        try:
            from app.services.survey_analyzer import SurveyAnalyzer
            
            self.analyzer = SurveyAnalyzer(json_data)
            self.analyzer.load_data()
            self.analyzer.extract_features()
            self.analyzer.create_feature_matrix()
            
            return True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")
    
    def train_model(self, n_clusters: Optional[int] = None, max_k: int = 6):
        """Entrena el modelo K-means - sin plots para mayor velocidad"""
        if self.analyzer is None:
            raise HTTPException(status_code=400, detail="No data loaded. Upload survey data first.")
        
        try:
            if n_clusters is None:
                n_clusters, silhouette = self.analyzer.find_optimal_clusters(max_k)
            else:
                silhouette = None
            
            cluster_labels = self.analyzer.perform_clustering(n_clusters)
            cluster_analysis = self.analyzer.analyze_clusters()
            self.analyzer.save_model(self.model_path)
            
            self._model_cache = None
            
            clean_analysis = clean_for_json(cluster_analysis)
            
            return {
                "success": True,
                "n_clusters": int(n_clusters),
                "silhouette_score": float(silhouette) if silhouette is not None else None,
                "cluster_analysis": clean_analysis,
                "model_saved": True
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")
    
    def _load_model_cached(self):
        """Carga el modelo con caché para mayor velocidad"""
        if self._model_cache is None:
            if not os.path.exists(self.model_path):
                raise HTTPException(status_code=400, detail="Model not trained. Train model first.")
            
            with open(self.model_path, 'rb') as f:
                self._model_cache = pickle.load(f)
        
        return self._model_cache
    
    def predict_cluster(self, user_features: UserFeatures):
        """Predice cluster - optimizado con caché"""
        try:
            model_data = self._load_model_cached()
            
            kmeans_model = model_data['kmeans_model']
            scaler = model_data['scaler']
            feature_columns = model_data['feature_columns']
            
            feature_vector = self._convert_user_features_to_vector(user_features, feature_columns)
            feature_vector_scaled = scaler.transform([feature_vector])
            cluster = kmeans_model.predict(feature_vector_scaled)[0]
            
            cluster_analysis = model_data.get('cluster_analysis', {})
            cluster_info = cluster_analysis.get(f'Cluster_{cluster}', {})
            
            return {
                "predicted_cluster": int(cluster),
                "cluster_info": cluster_info,
                "confidence": float(kmeans_model.score([feature_vector_scaled[0]]))
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error predicting cluster: {str(e)}")
    
    def _convert_user_features_to_vector(self, user_features: UserFeatures, feature_columns: List[str]):
        """Convierte características de usuario a vector de features"""
        feature_vector = []
        
        for col in feature_columns:
            if col == 'total_encuestas':
                feature_vector.append(user_features.total_encuestas)
            elif col == 'total_respuestas':
                feature_vector.append(user_features.total_respuestas)
            elif col == 'diversidad_respuestas':
                feature_vector.append(user_features.diversidad_respuestas)
            elif col.startswith('tipo_'):
                tipo = col.replace('tipo_', '').replace('_', ' ').title()
                feature_vector.append(user_features.tipos_respuesta.get(tipo, 0))
            elif col.startswith('canal_'):
                canal = col.replace('canal_', '').title()
                feature_vector.append(user_features.canales.get(canal, 0))
            elif col.startswith('tema_'):
                tema = col.replace('tema_', '')
                feature_vector.append(user_features.temas_encuesta.get(tema, 0))
            elif col.startswith('patron_'):
                patron = col.replace('patron_', '')
                feature_vector.append(user_features.patrones_respuesta.get(patron, 0))
            else:
                feature_vector.append(0)
        
        return feature_vector
    
    def reset_model(self):
        """Resetea el modelo y limpia datos"""
        self.analyzer = None
        self._model_cache = None
        
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
