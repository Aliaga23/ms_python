from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import pickle
import os
import json
import re
from typing import List, Dict
from collections import defaultdict

class KMeansService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.scaler = StandardScaler()
            cls._instance.kmeans = None
            cls._instance.feature_names = []
            cls._instance.is_trained = False
            cls._instance.model_path = "kmeans_model.pkl"
            cls._instance.category_keywords = {}
            cls._instance._load_categories()
        return cls._instance
    
    def _load_categories(self):
        categories_path = os.path.join('app', 'config', 'categories.json')
        try:
            with open(categories_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.category_keywords = data.get('categorias', {})
        except Exception as e:
            print(f"Warning: No se pudo cargar categories.json: {e}")
            self.category_keywords = {}
    
    def _detect_category(self, texto: str) -> str:
        texto_clean = texto.lower().strip()
        texto_clean = re.sub(r'[^\wáéíóúñü\s]', ' ', texto_clean)
        
        category_scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = 0
            for keyword in keywords:
                keyword_clean = keyword.lower()
                pattern = r'\b' + re.escape(keyword_clean) + r'\b'
                if re.search(pattern, texto_clean):
                    score += 2
                elif keyword_clean in texto_clean:
                    score += 1
            
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])[0]
            return best_category
        
        return "general"
    
    def analyze_survey(self, survey_data: Dict) -> Dict:
        encuesta = survey_data.get('encuesta', {})
        preguntas_meta = survey_data.get('preguntas', [])
        respuestas = survey_data.get('respuestas', [])
        
        if not respuestas:
            return {"error": "No hay respuestas en la encuesta"}
        
        texto_completo_encuesta = f"{encuesta.get('nombre', '')} {encuesta.get('descripcion', '')}"
        categoria_encuesta = self._detect_category(texto_completo_encuesta)
        
        entregas = defaultdict(lambda: defaultdict(list))
        
        for resp in respuestas:
            entrega_id = resp.get('entrega_id')
            pregunta_id = resp.get('pregunta_id')
            pregunta_texto = resp.get('pregunta_texto', '')
            valores_seleccionados = resp.get('valores_seleccionados', [])
            
            valores_numericos = []
            for val in valores_seleccionados:
                try:
                    valores_numericos.append(float(val.get('valor', 0)))
                except (ValueError, TypeError):
                    valores_numericos.append(0)
            
            if valores_numericos:
                promedio_valor = sum(valores_numericos) / len(valores_numericos)
            else:
                promedio_valor = 0
            
            entregas[entrega_id][pregunta_id] = {
                'pregunta': pregunta_texto,
                'valor': promedio_valor,
                'cantidad_respuestas': len(valores_numericos)
            }
        
        feature_matrix = []
        entrega_ids = []
        pregunta_features = {}
        pregunta_categorias = {}
        
        for entrega_id, preguntas in entregas.items():
            features = []
            entrega_ids.append(entrega_id)
            
            for pregunta_id, datos in sorted(preguntas.items()):
                features.append(datos['valor'])
                if pregunta_id not in pregunta_features:
                    pregunta_texto = datos['pregunta']
                    pregunta_features[pregunta_id] = pregunta_texto
                    categoria_pregunta = self._detect_category(pregunta_texto)
                    if categoria_pregunta == "general":
                        categoria_pregunta = categoria_encuesta
                    pregunta_categorias[pregunta_id] = categoria_pregunta
            
            feature_matrix.append(features)
        
        self.feature_names = [pregunta_features[pid] for pid in sorted(pregunta_features.keys())]
        pregunta_ids_sorted = sorted(pregunta_features.keys())
        categorias_por_pregunta = [pregunta_categorias[pid] for pid in pregunta_ids_sorted]
        
        if len(feature_matrix) < 2:
            return {"error": "Se necesitan al menos 2 entregas para clustering"}
        
        X = np.array(feature_matrix)
        X_scaled = self.scaler.fit_transform(X)
        
        n_entregas = len(entregas)
        max_clusters = min(5, n_entregas - 1)
        
        best_k = 2
        best_silhouette = -1
        silhouette_scores = {}
        
        for k in range(2, max_clusters + 1):
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans_temp.fit_predict(X_scaled)
            silhouette = silhouette_score(X_scaled, labels)
            silhouette_scores[k] = float(silhouette)
            
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_k = k
        
        self.kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        self.is_trained = True
        
        clusters_info = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_labels):
            clusters_info[int(cluster_id)].append({
                'entrega_id': entrega_ids[idx],
                'valores': feature_matrix[idx]
            })
        
        cluster_analysis = {}
        for cluster_id, entregas_list in clusters_info.items():
            valores_cluster = [e['valores'] for e in entregas_list]
            promedio = np.mean(valores_cluster, axis=0).tolist()
            
            preguntas_con_categoria = []
            for i in range(len(promedio)):
                preguntas_con_categoria.append({
                    'pregunta': self.feature_names[i],
                    'categoria': categorias_por_pregunta[i],
                    'promedio': round(promedio[i], 2)
                })
            
            categorias_cluster = {}
            for pcc in preguntas_con_categoria:
                cat = pcc['categoria']
                if cat not in categorias_cluster:
                    categorias_cluster[cat] = []
                categorias_cluster[cat].append({
                    'pregunta': pcc['pregunta'],
                    'promedio': pcc['promedio']
                })
            
            cluster_analysis[f'Cluster_{cluster_id}'] = {
                'cantidad_entregas': len(entregas_list),
                'entregas': [e['entrega_id'] for e in entregas_list],
                'preguntas_con_categorias': preguntas_con_categoria,
                'analisis_por_categoria': categorias_cluster,
                'interpretacion': self._interpretar_cluster(promedio)
            }
        
        result = {
            "success": True,
            "encuesta": {
                "nombre": encuesta.get('nombre', 'Sin nombre'),
                "descripcion": encuesta.get('descripcion', ''),
                "categoria_detectada": categoria_encuesta
            },
            "total_entregas": n_entregas,
            "total_preguntas": len(self.feature_names),
            "preguntas_con_categorias": [
                {
                    'pregunta': self.feature_names[i],
                    'categoria': categorias_por_pregunta[i]
                }
                for i in range(len(self.feature_names))
            ],
            "n_clusters": best_k,
            "silhouette_score": float(best_silhouette),
            "silhouette_scores": silhouette_scores,
            "clusters": cluster_analysis
        }
        
        return result
    
    def _interpretar_cluster(self, promedios: List[float]) -> str:
        promedio_general = np.mean(promedios)
        
        if promedio_general >= 4:
            return "Alto rendimiento - Valores muy positivos"
        elif promedio_general >= 3:
            return "Rendimiento medio-alto - Buenos resultados"
        elif promedio_general >= 2:
            return "Rendimiento medio - Resultados aceptables"
        else:
            return "Bajo rendimiento - Requiere atención"
    
    def reset_model(self):
        self.is_trained = False
        self.kmeans = None
        self.feature_names = []
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
