from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import os
import json
import re
from typing import List, Dict, Tuple
from app.models.text_analysis import (
    AnomalyDetectionResult,
    ClassificationResult,
    TextAnalysisResponse
)

class TextAnalysisService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            
            # Stop words en español
            cls._instance.spanish_stop_words = [
                'el', 'la', 'de', 'en', 'a', 'los', 'las', 'se', 'con', 'para',
                'un', 'una', 'por', 'al', 'del', 'como', 'es', 'su', 'sus',
                'que', 'y', 'o', 'si', 'no', 'esto', 'esta', 'estos', 'estas',
                'fue', 'son', 'sobre', 'entre', 'hasta', 'desde', 'durante'
            ]
            
            # Palabras de sentimiento positivo (generales)
            cls._instance.palabras_positivas = [
                'excelente', 'bueno', 'buena', 'increíble', 'perfecto', 'perfecta',
                'delicioso', 'deliciosa', 'agradable', 'rico', 'rica', 'suave',
                'intenso', 'intensa', 'vibrante', 'fresco', 'fresca', 'aromático',
                'aromática', 'equilibrado', 'equilibrada', 'brillante', 'encantador',
                'encantadora', 'elegante', 'sofisticado', 'sofisticada', 'premium',
                'genial', 'espectacular', 'maravilloso', 'maravillosa', 'fantástico',
                'fantástica', 'hermoso', 'hermosa', 'jugoso', 'jugosa', 'cremoso',
                'cremosa', 'sutil', 'complejo', 'compleja', 'refinado', 'refinada',
                'extraordinario', 'extraordinaria', 'memorable', 'sorprendente',
                'destacado', 'destacada', 'superior', 'óptimo', 'óptima', 'ideal',
                'me gusta', 'me encanta', 'recomiendo', 'amor', 'amo', 'fascina',
                'satisfecho', 'satisfactorio', 'eficiente', 'rápido', 'oportuno',
                'confiable', 'profesional', 'atento', 'amable', 'cumple', 'supera'
            ]
            
            # Palabras de sentimiento negativo (generales)
            cls._instance.palabras_negativas = [
                'malo', 'mala', 'horrible', 'terrible', 'desagradable', 'feo',
                'fea', 'pésimo', 'pésima', 'amargo', 'amarga', 'ácido', 'ácida',
                'insípido', 'insípida', 'soso', 'sosa', 'débil', 'pobre', 'sin',
                'falta', 'carece', 'deficiente', 'mediocre', 'decepcionante',
                'mal', 'peor', 'nada', 'ninguna', 'ningún', 'poco', 'apenas',
                'defecto', 'problema', 'falló', 'fallado', 'rancio', 'rancia',
                'pasado', 'pasada', 'excesivo', 'excesiva', 'demasiado', 
                'desbalanceado', 'desbalanceada', 'artificial', 'no me gusta', 
                'no recomiendo', 'evitar', 'lento', 'tardío', 'retraso', 'demora',
                'incompleto', 'incumplimiento', 'error', 'falla', 'daño', 'dañado',
                'roto', 'fracaso', 'insatisfecho', 'insatisfactorio', 'caro', 'costoso'
            ]
            
            # Palabras negativas CONTEXTUALES (solo negativas en categorías específicas)
            cls._instance.palabras_negativas_contextuales = {
                'calidad_sensorial_producto': ['oxidado', 'oxidada', 'químico', 'química', 
                                                'quemado', 'quemada', 'avinagrado', 'avinagrada'],
                'operaciones_mantenimiento': ['oxidado', 'oxidada', 'corroído', 'corroída'],
                'logistica_entregas': ['retrasado', 'retrasada', 'tardío', 'tardía'],
                'rendimiento': ['lento', 'lenta', 'ineficiente']
            }
            
            cls._instance.vectorizer = TfidfVectorizer(
                max_features=500,
                ngram_range=(1, 3), 
                min_df=1,  
                max_df=0.8,
                stop_words=cls._instance.spanish_stop_words,
                sublinear_tf=True,
                strip_accents='unicode'  
            )
            
            cls._instance.isolation_forest = IsolationForest(
                contamination=0.1,
                random_state=42,
                max_samples=256,
                n_estimators=150,
                bootstrap=True
            )
            
            cls._instance.svm_classifier = None
            cls._instance.label_encoder = LabelEncoder()
            cls._instance.is_trained = False
            cls._instance.model_path = "text_analysis_model.pkl"
            
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
    
    def _extract_category_from_question(self, pregunta: str) -> str:
        pregunta_clean = pregunta.lower().strip()
        pregunta_clean = re.sub(r'[^\wáéíóúñü\s]', ' ', pregunta_clean)
        
        category_scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                keyword_clean = keyword.lower()
                pattern = r'\b' + re.escape(keyword_clean) + r'\b'
                if re.search(pattern, pregunta_clean):
                    score += 2
                    matched_keywords.append(keyword)
                elif keyword_clean in pregunta_clean:
                    score += 1
                    matched_keywords.append(keyword)
            
            if score > 0:
                category_scores[category] = (score, len(matched_keywords))
        
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: (x[1][0], x[1][1]))[0]
            return best_category
        
        stop_words = ['describa', 'indique', 'anote', 'mencione', 'detalle', 'especifique',
                      'señale', 'explique', 'comente', 'reporte', 'cuéntenos', 'qué', 'con',
                      'cómo', 'cuál', 'dónde'] + self.spanish_stop_words
        
        words = [w for w in pregunta_clean.split() if w not in stop_words and len(w) > 3]
        
        if words:
            return "_".join(words[:2]).title()
        
        return "general"
    
    def _analyze_sentiment(self, texto: str, categoria: str = None) -> Tuple[str, float]:
        texto_lower = texto.lower()
        
        positivos = sum(1 for palabra in self.palabras_positivas if palabra in texto_lower)
        
        negativos = sum(1 for palabra in self.palabras_negativas if palabra in texto_lower)
        
        if categoria and categoria in self.palabras_negativas_contextuales:
            negativos += sum(1 for palabra in self.palabras_negativas_contextuales[categoria] 
                           if palabra in texto_lower)
        
        total_palabras = len(texto_lower.split())
        if total_palabras == 0:
            return 'neutro', 0.0
        
        score = (positivos - negativos) / max(total_palabras, 1)
        score = max(-1.0, min(1.0, score * 10))
        
        if score > 0.15:
            return 'positivo', score
        elif score < -0.15:
            return 'negativo', score
        else:
            return 'neutro', score
    
    def train_model(self, respuestas: List[Dict[str, str]]) -> Dict:
        textos = [r['texto_respuesta'] for r in respuestas]
        preguntas = [r.get('pregunta_texto', '') for r in respuestas]
        
        categorias = [self._extract_category_from_question(p) for p in preguntas]
        
        X = self.vectorizer.fit_transform(textos)
        
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            max_samples=min(256, len(textos)),
            n_estimators=150,
            bootstrap=True
        )
        self.isolation_forest.fit(X.toarray())
        
        if len(set(categorias)) > 1:
            y = self.label_encoder.fit_transform(categorias)
            
            base_svm = LinearSVC(random_state=42, max_iter=2000, dual='auto')
            self.svm_classifier = CalibratedClassifierCV(base_svm, cv=3)
            self.svm_classifier.fit(X, y)
        
        self.is_trained = True
        
        self._save_model()
        
        return {
            "success": True,
            "total_respuestas": len(respuestas),
            "categorias": list(set(categorias)),
            "features_extracted": X.shape[1]
        }
    
    def analyze_responses(self, respuestas: List[Dict[str, str]]) -> TextAnalysisResponse:
        if not self.is_trained:
            self.train_model(respuestas)
        
        textos = [r['texto_respuesta'] for r in respuestas]
        respuesta_ids = [r.get('respuesta_id', f'resp_{i}') for i, r in enumerate(respuestas)]
        preguntas = [r.get('pregunta_texto', '') for r in respuestas]
        
        X = self.vectorizer.transform(textos)
        
        anomaly_predictions = self.isolation_forest.predict(X.toarray())
        anomaly_scores = self.isolation_forest.score_samples(X.toarray())
        
        anomalias = []
        for i, (pred, score) in enumerate(zip(anomaly_predictions, anomaly_scores)):
            texto = textos[i]
            palabras = len(texto.split())
            
            palabras_validas = ['no', 'ninguna', 'nada', 'mal', 'malo', 'mala', 'feo', 
                               'desagradable', 'horrible', 'terrible', 'poco', 'mucho',
                               'demasiado', 'apenas', 'sin', 'falta']
            texto_lower = texto.lower()
            tiene_feedback_valido = any(palabra in texto_lower for palabra in palabras_validas)
            
            is_anomaly_length_short = palabras < 8
            is_anomaly_length_long = palabras > 100
            is_anomaly_score = score < -0.35
            
            if (is_anomaly_length_short and is_anomaly_score and not tiene_feedback_valido) or is_anomaly_length_long:
                anomalias.append(AnomalyDetectionResult(
                    respuesta_id=respuesta_ids[i],
                    texto_respuesta=texto,
                    is_anomaly=True,
                    anomaly_score=float(score),
                    pregunta_texto=preguntas[i] if i < len(preguntas) else None
                ))
        
        clasificaciones = []
        categorias_count = {}
        sentimiento_count = {'positivo': 0, 'negativo': 0, 'neutro': 0}
        sentimiento_por_categoria = {}
        categorias_vistas = {}
        
        if self.svm_classifier is not None:
            predictions = self.svm_classifier.predict(X)
            probabilities = self.svm_classifier.predict_proba(X)
            
            for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
                categoria = self.label_encoder.inverse_transform([pred])[0]
                confianza = float(np.max(proba))
                
                sentimiento, sentimiento_score = self._analyze_sentiment(textos[i], categoria)
                sentimiento_count[sentimiento] += 1
                
                if categoria not in sentimiento_por_categoria:
                    sentimiento_por_categoria[categoria] = {'positivo': 0, 'negativo': 0, 'neutro': 0}
                sentimiento_por_categoria[categoria][sentimiento] += 1
                
                if categoria not in categorias_vistas:
                    clasificaciones.append(ClassificationResult(
                        respuesta_id=respuesta_ids[i],
                        texto_respuesta=textos[i][:100] + "..." if len(textos[i]) > 100 else textos[i],
                        categoria_predicha=categoria,
                        confianza=confianza,
                        sentimiento=sentimiento,
                        sentimiento_score=sentimiento_score,
                        pregunta_texto=preguntas[i] if i < len(preguntas) else None
                    ))
                    categorias_vistas[categoria] = True
                
                categorias_count[categoria] = categorias_count.get(categoria, 0) + 1
        
        return TextAnalysisResponse(
            total_respuestas=len(respuestas),
            anomalias_detectadas=len(anomalias),
            anomalias=anomalias,
            ejemplos_por_categoria=clasificaciones,
            categorias_encontradas=categorias_count,
            sentimiento_resumen=sentimiento_count,
            sentimiento_por_categoria=sentimiento_por_categoria
        )
    
    def predict_single(self, texto: str) -> Dict:
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        X = self.vectorizer.transform([texto])
        
        is_anomaly = self.isolation_forest.predict(X.toarray())[0] == -1
        anomaly_score = float(self.isolation_forest.score_samples(X.toarray())[0])
        
        categoria = None
        confianza = None
        sentimiento = None
        sentimiento_score = None
        
        if self.svm_classifier is not None:
            pred = self.svm_classifier.predict(X)[0]
            proba = self.svm_classifier.predict_proba(X)[0]
            categoria = self.label_encoder.inverse_transform([pred])[0]
            confianza = float(np.max(proba))
            
            sentimiento, sentimiento_score = self._analyze_sentiment(texto, categoria)
        
        return {
            "texto": texto,
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": anomaly_score,
            "categoria_predicha": categoria,
            "confianza": confianza,
            "sentimiento": sentimiento,
            "sentimiento_score": float(sentimiento_score) if sentimiento_score else None
        }
    
    def _save_model(self):
        model_data = {
            'vectorizer': self.vectorizer,
            'isolation_forest': self.isolation_forest,
            'svm_classifier': self.svm_classifier,
            'label_encoder': self.label_encoder,
            'is_trained': self.is_trained
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data['vectorizer']
            self.isolation_forest = model_data['isolation_forest']
            self.svm_classifier = model_data['svm_classifier']
            self.label_encoder = model_data['label_encoder']
            self.is_trained = model_data['is_trained']
            
            return True
        return False
    
    def reset_model(self):
        self.is_trained = False
        self.svm_classifier = None
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
