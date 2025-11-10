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
    
    def _analyze_flat_format(self, respuestas_list: List[Dict]) -> Dict:
        if not respuestas_list:
            return {"error": "No hay respuestas en la lista"}
        
        primera_respuesta = respuestas_list[0]
        encuesta_nombre = primera_respuesta.get('encuesta_nombre', 'Sin nombre')
        
        categoria_encuesta = self._detect_category(encuesta_nombre)
        
        # Agrupar respuestas por encuesta_id
        encuestas_agrupadas = defaultdict(lambda: defaultdict(dict))
        opciones_por_pregunta = defaultdict(list)
        
        # Primero, recopilar todas las opciones disponibles por pregunta
        for resp in respuestas_list:
            pregunta_texto = resp.get('pregunta_texto', '')
            opciones_disponibles = resp.get('opciones_disponibles', [])
            
            # Solo actualizar si encontramos opciones y no las tenemos ya
            if pregunta_texto and opciones_disponibles and pregunta_texto not in opciones_por_pregunta:
                opciones_por_pregunta[pregunta_texto] = opciones_disponibles
        
        # Ahora procesar todas las respuestas agrupándolas por destinatario
        for resp in respuestas_list:
            destinatario_nombre = resp.get('destinatario_nombre', resp.get('user_id', 'unknown'))
            pregunta_texto = resp.get('pregunta_texto', '')
            encuesta_nombre = resp.get('encuesta_nombre', '')
            opciones_seleccionadas = resp.get('opciones_seleccionadas', [])
            opciones_disponibles = resp.get('opciones_disponibles', [])
            
            # Crear una clave única para la pregunta incluyendo la encuesta
            pregunta_clave = f"[{encuesta_nombre}] {pregunta_texto}"
            
            # Asegurar que tenemos las opciones para esta pregunta
            if pregunta_clave and opciones_disponibles and pregunta_clave not in opciones_por_pregunta:
                opciones_por_pregunta[pregunta_clave] = opciones_disponibles
            
            valores_numericos = []
            textos_seleccionados = []
            
            # Procesar opciones seleccionadas
            if opciones_seleccionadas:
                for opcion in opciones_seleccionadas:
                    try:
                        valor_str = str(opcion.get('valor', '0')).strip()
                        if valor_str:
                            valores_numericos.append(float(valor_str))
                            textos_seleccionados.append(opcion.get('texto', ''))
                    except (ValueError, TypeError):
                        valores_numericos.append(0)
                        textos_seleccionados.append(opcion.get('texto', ''))
            
            # Solo procesar si hay respuestas válidas (no agregar 0s artificiales)
            if valores_numericos and any(v > 0 for v in valores_numericos):
                promedio_valor = sum(valores_numericos) / len(valores_numericos)
                
                # Agrupar por destinatario y pregunta_clave
                if pregunta_clave in encuestas_agrupadas[destinatario_nombre]:
                    # Si ya existe una respuesta para esta pregunta de este destinatario, promediar
                    valor_anterior = encuestas_agrupadas[destinatario_nombre][pregunta_clave]['valor']
                    count_anterior = encuestas_agrupadas[destinatario_nombre][pregunta_clave]['cantidad_respuestas']
                    nuevo_promedio = ((valor_anterior * count_anterior) + promedio_valor) / (count_anterior + 1)
                    
                    encuestas_agrupadas[destinatario_nombre][pregunta_clave] = {
                        'valor': nuevo_promedio,
                        'cantidad_respuestas': count_anterior + 1,
                        'opciones_seleccionadas': encuestas_agrupadas[destinatario_nombre][pregunta_clave]['opciones_seleccionadas'] + textos_seleccionados
                    }
                else:
                    encuestas_agrupadas[destinatario_nombre][pregunta_clave] = {
                        'valor': promedio_valor,
                        'cantidad_respuestas': 1,
                        'opciones_seleccionadas': textos_seleccionados
                    }
        
        # Filtrar destinatarios que tienen pocas respuestas (menos del 30% de las preguntas)
        todas_las_preguntas = set()
        for destinatario_dict in encuestas_agrupadas.values():
            todas_las_preguntas.update(destinatario_dict.keys())
        
        min_respuestas = max(1, len(todas_las_preguntas) // 3)  # Al menos 30% de respuestas
        
        destinatarios_filtrados = {}
        for destinatario, preguntas in encuestas_agrupadas.items():
            if len(preguntas) >= min_respuestas:
                destinatarios_filtrados[destinatario] = preguntas
        
        if len(destinatarios_filtrados) < 2:
            return {"error": f"Después del filtrado, solo hay {len(destinatarios_filtrados)} destinatarios con suficientes respuestas. Se necesitan al menos 2."}
        
        encuestas_agrupadas = destinatarios_filtrados
        
        feature_matrix = []
        entrega_ids = []
        pregunta_features = []
        pregunta_categorias = []
        
        preguntas_unicas = sorted(set(
            pregunta_texto 
            for destinatario_dict in encuestas_agrupadas.values() 
            for pregunta_texto in destinatario_dict.keys()
        ))
        
        for pregunta in preguntas_unicas:
            pregunta_features.append(pregunta)
            # Extraer el nombre de la encuesta de la pregunta
            if pregunta.startswith('['):
                encuesta_parte = pregunta.split(']')[0][1:]  # Extraer entre []
                categoria_pregunta = self._detect_category(encuesta_parte)
            else:
                categoria_pregunta = self._detect_category(pregunta)
            if categoria_pregunta == "general":
                categoria_pregunta = categoria_encuesta
            pregunta_categorias.append(categoria_pregunta)
        
        for destinatario, preguntas in encuestas_agrupadas.items():
            features = []
            entrega_ids.append(destinatario)
            
            for pregunta in pregunta_features:
                if pregunta in preguntas:
                    features.append(preguntas[pregunta]['valor'])
                else:
                    features.append(0)
            
            feature_matrix.append(features)
        
        self.feature_names = pregunta_features
        
        if len(feature_matrix) < 2:
            return {"error": "Se necesitan al menos 2 destinatarios para clustering"}
        
        X = np.array(feature_matrix)
        
        # Solo escalar si hay varianza en los datos
        if np.std(X) > 0:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        n_destinatarios = len(encuestas_agrupadas)
        max_clusters = min(4, n_destinatarios - 1)  # Reducir máximo de clusters
        
        best_k = 2
        best_silhouette = -1
        silhouette_scores = {}
        
        # Solo hacer clustering si tenemos suficientes datos
        if n_destinatarios >= 3:
            for k in range(2, max_clusters + 1):
                kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans_temp.fit_predict(X_scaled)
                
                # Verificar que hay al menos 2 puntos por cluster
                unique_labels, counts = np.unique(labels, return_counts=True)
                if len(unique_labels) == k and all(counts >= 1):
                    silhouette = silhouette_score(X_scaled, labels)
                    silhouette_scores[k] = float(silhouette)
                    
                    if silhouette > best_silhouette:
                        best_silhouette = silhouette
                        best_k = k
        
        # Si no se encontró un buen silhouette score, usar 2 clusters por defecto
        if best_silhouette < 0.1:
            best_k = 2
        
        self.kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        self.is_trained = True
        
        clusters_info = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_labels):
            clusters_info[int(cluster_id)].append({
                'destinatario': entrega_ids[idx],
                'valores': feature_matrix[idx]
            })
        
        cluster_analysis = {}
        for cluster_id, destinatarios_list in clusters_info.items():
            valores_cluster = [e['valores'] for e in destinatarios_list]
            valores_array = np.array(valores_cluster)
            
            promedio = np.mean(valores_array, axis=0)
            desviacion = np.std(valores_array, axis=0)
            minimos = np.min(valores_array, axis=0)
            maximos = np.max(valores_array, axis=0)
            mediana = np.median(valores_array, axis=0)
            
            preguntas_con_categoria = []
            for i in range(len(promedio)):
                pregunta = self.feature_names[i]
                opciones_info = opciones_por_pregunta.get(pregunta, [])
                valor_promedio = float(promedio[i])
                
                interpretacion_texto = self._interpretar_valor_con_opciones(valor_promedio, opciones_info)
                
                preguntas_con_categoria.append({
                    'pregunta': pregunta,
                    'categoria': pregunta_categorias[i],
                    'promedio': round(valor_promedio, 2),
                    'interpretacion': interpretacion_texto,
                    'desviacion_estandar': round(float(desviacion[i]), 2),
                    'minimo': round(float(minimos[i]), 2),
                    'maximo': round(float(maximos[i]), 2),
                    'mediana': round(float(mediana[i]), 2),
                    'opciones_disponibles': opciones_info
                })
            
            categorias_cluster = {}
            categorias_stats = {}
            
            for pcc in preguntas_con_categoria:
                cat = pcc['categoria']
                if cat not in categorias_cluster:
                    categorias_cluster[cat] = []
                    categorias_stats[cat] = {
                        'promedios': [],
                        'desviaciones': [],
                        'conteo_preguntas': 0
                    }
                
                categorias_cluster[cat].append({
                    'pregunta': pcc['pregunta'],
                    'promedio': pcc['promedio'],
                    'desviacion_estandar': pcc['desviacion_estandar'],
                    'rango': [pcc['minimo'], pcc['maximo']]
                })
                
                categorias_stats[cat]['promedios'].append(pcc['promedio'])
                categorias_stats[cat]['desviaciones'].append(pcc['desviacion_estandar'])
                categorias_stats[cat]['conteo_preguntas'] += 1
            
            resumen_categorias = {}
            for cat, stats in categorias_stats.items():
                resumen_categorias[cat] = {
                    'promedio_general': round(float(np.mean(stats['promedios'])), 2),
                    'desviacion_promedio': round(float(np.mean(stats['desviaciones'])), 2),
                    'total_preguntas': stats['conteo_preguntas']
                }
            
            promedio_general_cluster = float(np.mean(promedio))
            desviacion_general_cluster = float(np.std(promedio))
            
            cluster_analysis[f'Cluster_{cluster_id}'] = {
                'id': cluster_id,
                'cantidad_destinatarios': len(destinatarios_list),
                'porcentaje_total': round((len(destinatarios_list) / len(entrega_ids)) * 100, 2),
                'destinatarios': [e['destinatario'] for e in destinatarios_list],
                'estadisticas_generales': {
                    'promedio_general': round(promedio_general_cluster, 2),
                    'desviacion_general': round(desviacion_general_cluster, 2),
                    'promedio_minimo': round(float(np.min(promedio)), 2),
                    'promedio_maximo': round(float(np.max(promedio)), 2)
                },
                'preguntas_detalladas': preguntas_con_categoria,
                'analisis_por_categoria': categorias_cluster,
                'resumen_categorias': resumen_categorias,
                'interpretacion': self._interpretar_cluster(promedio.tolist()),
                'centroide': [round(float(x), 3) for x in self.kmeans.cluster_centers_[cluster_id]]
            }
        
        comparativa_clusters = []
        for cluster_name, cluster_data in cluster_analysis.items():
            comparativa_clusters.append({
                'cluster_id': cluster_data['id'],
                'cluster_nombre': cluster_name,
                'cantidad_destinatarios': cluster_data['cantidad_destinatarios'],
                'porcentaje': cluster_data['porcentaje_total'],
                'promedio_general': cluster_data['estadisticas_generales']['promedio_general'],
                'desviacion_general': cluster_data['estadisticas_generales']['desviacion_general']
            })
        
        comparativa_clusters.sort(key=lambda x: x['promedio_general'], reverse=True)
        
        estadisticas_globales = {
            'promedio_todas_respuestas': round(float(np.mean(X)), 2),
            'desviacion_todas_respuestas': round(float(np.std(X)), 2),
            'valor_minimo_global': round(float(np.min(X)), 2),
            'valor_maximo_global': round(float(np.max(X)), 2),
            'mediana_global': round(float(np.median(X)), 2)
        }
        
        distribucion_por_pregunta = []
        for i, pregunta in enumerate(self.feature_names):
            valores_pregunta = X[:, i]
            opciones_info = opciones_por_pregunta.get(pregunta, [])
            
            distribucion_por_pregunta.append({
                'pregunta': pregunta,
                'categoria': pregunta_categorias[i],
                'promedio': round(float(np.mean(valores_pregunta)), 2),
                'desviacion': round(float(np.std(valores_pregunta)), 2),
                'minimo': round(float(np.min(valores_pregunta)), 2),
                'maximo': round(float(np.max(valores_pregunta)), 2),
                'mediana': round(float(np.median(valores_pregunta)), 2),
                'opciones_disponibles': opciones_info
            })
        
        result = {
            "success": True,
            "total_destinatarios": n_destinatarios,
            "n_clusters": best_k,
            "silhouette_score": round(float(best_silhouette), 3),
            "clusters_summary": self._generate_clusters_summary(cluster_analysis, pregunta_categorias)
        }
        
        return result
    
    def _generate_clusters_summary(self, cluster_analysis, pregunta_categorias):
        """Genera un resumen simplificado de clusters para GPT"""
        clusters_summary = []
        
        # Mapear categorías a nombres más amigables
        categoria_nombres = {
            "operaciones_mantenimiento": "Mantenimiento Predictivo",
            "rendimiento": "Optimización de Rendimiento", 
            "seguridad_cumplimiento": "Seguridad y Cumplimiento",
            "calidad_conformidad": "Control de Calidad"
        }
        
        for cluster_name, cluster_data in cluster_analysis.items():
            cluster_id = cluster_data['id']
            cantidad = cluster_data['cantidad_destinatarios']
            porcentaje = cluster_data['porcentaje_total']
            
            # Identificar la especialidad principal del cluster
            especialidades = {}
            respuestas_validas = []
            
            for pregunta_detail in cluster_data['preguntas_detalladas']:
                if pregunta_detail['promedio'] > 0:  # Solo considerar respuestas válidas
                    categoria = pregunta_detail['categoria']
                    valor = pregunta_detail['promedio']
                    interpretacion = pregunta_detail['interpretacion']
                    
                    if categoria not in especialidades:
                        especialidades[categoria] = []
                    
                    especialidades[categoria].append({
                        'pregunta': pregunta_detail['pregunta'].split('] ')[1] if '] ' in pregunta_detail['pregunta'] else pregunta_detail['pregunta'],
                        'valor': valor,
                        'interpretacion': interpretacion
                    })
                    
                    respuestas_validas.append({
                        'categoria': categoria,
                        'pregunta': pregunta_detail['pregunta'].split('] ')[1] if '] ' in pregunta_detail['pregunta'] else pregunta_detail['pregunta'],
                        'valor': valor,
                        'interpretacion': interpretacion
                    })
            
            # Determinar especialidad principal
            especialidad_principal = max(especialidades.keys(), key=lambda x: len(especialidades[x])) if especialidades else "General"
            nombre_cluster = categoria_nombres.get(especialidad_principal, especialidad_principal.title())
            
            cluster_summary = {
                "cluster_id": cluster_id,
                "nombre": nombre_cluster,
                "descripcion": f"Grupo especializado en {nombre_cluster.lower()}",
                "cantidad_personas": cantidad,
                "porcentaje": round(porcentaje, 1),
                "especialidad_principal": especialidad_principal,
                "respuestas_destacadas": respuestas_validas[:3],  # Top 3 respuestas más relevantes
                "valores_para_graficas": {
                    "promedio_general": cluster_data['estadisticas_generales']['promedio_general'],
                    "rango_min": cluster_data['estadisticas_generales']['promedio_minimo'],
                    "rango_max": cluster_data['estadisticas_generales']['promedio_maximo']
                }
            }
            
            clusters_summary.append(cluster_summary)
        
        # Ordenar por promedio general (mayor a menor)
        clusters_summary.sort(key=lambda x: x['valores_para_graficas']['promedio_general'], reverse=True)
        
        return clusters_summary
    
    def analyze_survey(self, survey_data: Dict) -> Dict:
        if isinstance(survey_data, list):
            return self._analyze_flat_format(survey_data)
        
        encuesta = survey_data.get('encuesta', {})
        preguntas_meta = survey_data.get('preguntas', [])
        respuestas = survey_data.get('respuestas', [])
        
        if not respuestas:
            return {"error": "No hay respuestas en la encuesta"}
        
        texto_completo_encuesta = f"{encuesta.get('nombre', '')} {encuesta.get('descripcion', '')}"
        categoria_encuesta = self._detect_category(texto_completo_encuesta)
        
        # Mapear opciones por pregunta_texto y pregunta_id
        opciones_por_pregunta_id = {}
        opciones_por_pregunta_texto = {}
        for pregunta_meta in preguntas_meta:
            pregunta_id = pregunta_meta.get('pregunta_id')
            pregunta_texto = pregunta_meta.get('pregunta_texto', '')
            # Intentar múltiples nombres de campo para opciones
            opciones = pregunta_meta.get('opciones_disponibles', pregunta_meta.get('opciones', []))
            
            if opciones:
                if pregunta_id:
                    opciones_por_pregunta_id[pregunta_id] = opciones
                if pregunta_texto:
                    opciones_por_pregunta_texto[pregunta_texto] = opciones
        
        entregas = defaultdict(lambda: defaultdict(dict))
        pregunta_id_a_texto = {}
        
        for resp in respuestas:
            entrega_id = resp.get('entrega_id')
            pregunta_id = resp.get('pregunta_id')
            pregunta_texto = resp.get('pregunta_texto', '')
            valores_seleccionados = resp.get('valores_seleccionados', [])
            
            # Extraer opciones desde la respuesta si están disponibles
            opciones_en_respuesta = resp.get('opciones_disponibles', [])
            if opciones_en_respuesta and pregunta_id not in opciones_por_pregunta_id:
                opciones_por_pregunta_id[pregunta_id] = opciones_en_respuesta
                if pregunta_texto:
                    opciones_por_pregunta_texto[pregunta_texto] = opciones_en_respuesta
            
            pregunta_id_a_texto[pregunta_id] = pregunta_texto
            
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
            
            # Buscar opciones por pregunta_id o por pregunta_texto
            opciones = opciones_por_pregunta_id.get(pregunta_id, opciones_por_pregunta_texto.get(pregunta_texto, []))
            
            entregas[entrega_id][pregunta_id] = {
                'pregunta': pregunta_texto,
                'valor': promedio_valor,
                'cantidad_respuestas': len(valores_numericos),
                'opciones_disponibles': opciones
            }
        
        feature_matrix = []
        entrega_ids = []
        pregunta_features = {}
        pregunta_categorias = {}
        pregunta_opciones = {}
        
        for entrega_id, preguntas in entregas.items():
            features = []
            entrega_ids.append(entrega_id)
            
            for pregunta_id, datos in sorted(preguntas.items()):
                features.append(datos['valor'])
                if pregunta_id not in pregunta_features:
                    pregunta_texto = datos['pregunta']
                    pregunta_features[pregunta_id] = pregunta_texto
                    pregunta_opciones[pregunta_texto] = datos.get('opciones_disponibles', [])
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
            valores_array = np.array(valores_cluster)
            
            promedio = np.mean(valores_array, axis=0)
            desviacion = np.std(valores_array, axis=0)
            minimos = np.min(valores_array, axis=0)
            maximos = np.max(valores_array, axis=0)
            mediana = np.median(valores_array, axis=0)
            
            preguntas_con_categoria = []
            for i in range(len(promedio)):
                pregunta = self.feature_names[i]
                opciones_info = pregunta_opciones.get(pregunta, [])
                valor_promedio = float(promedio[i])
                
                interpretacion_texto = self._interpretar_valor_con_opciones(valor_promedio, opciones_info)
                
                preguntas_con_categoria.append({
                    'pregunta': pregunta,
                    'categoria': categorias_por_pregunta[i],
                    'promedio': round(valor_promedio, 2),
                    'interpretacion': interpretacion_texto,
                    'desviacion_estandar': round(float(desviacion[i]), 2),
                    'minimo': round(float(minimos[i]), 2),
                    'maximo': round(float(maximos[i]), 2),
                    'mediana': round(float(mediana[i]), 2),
                    'opciones_disponibles': opciones_info
                })
            
            categorias_cluster = {}
            categorias_stats = {}
            
            for pcc in preguntas_con_categoria:
                cat = pcc['categoria']
                if cat not in categorias_cluster:
                    categorias_cluster[cat] = []
                    categorias_stats[cat] = {
                        'promedios': [],
                        'desviaciones': [],
                        'conteo_preguntas': 0
                    }
                
                categorias_cluster[cat].append({
                    'pregunta': pcc['pregunta'],
                    'promedio': pcc['promedio'],
                    'desviacion_estandar': pcc['desviacion_estandar'],
                    'rango': [pcc['minimo'], pcc['maximo']]
                })
                
                categorias_stats[cat]['promedios'].append(pcc['promedio'])
                categorias_stats[cat]['desviaciones'].append(pcc['desviacion_estandar'])
                categorias_stats[cat]['conteo_preguntas'] += 1
            
            resumen_categorias = {}
            for cat, stats in categorias_stats.items():
                resumen_categorias[cat] = {
                    'promedio_general': round(float(np.mean(stats['promedios'])), 2),
                    'desviacion_promedio': round(float(np.mean(stats['desviaciones'])), 2),
                    'total_preguntas': stats['conteo_preguntas']
                }
            
            promedio_general_cluster = float(np.mean(promedio))
            desviacion_general_cluster = float(np.std(promedio))
            
            cluster_analysis[f'Cluster_{cluster_id}'] = {
                'id': cluster_id,
                'cantidad_entregas': len(entregas_list),
                'porcentaje_total': round((len(entregas_list) / len(entrega_ids)) * 100, 2),
                'entregas': [e['entrega_id'] for e in entregas_list],
                'estadisticas_generales': {
                    'promedio_general': round(promedio_general_cluster, 2),
                    'desviacion_general': round(desviacion_general_cluster, 2),
                    'promedio_minimo': round(float(np.min(promedio)), 2),
                    'promedio_maximo': round(float(np.max(promedio)), 2)
                },
                'preguntas_detalladas': preguntas_con_categoria,
                'analisis_por_categoria': categorias_cluster,
                'resumen_categorias': resumen_categorias,
                'interpretacion': self._interpretar_cluster(promedio.tolist()),
                'centroide': [round(float(x), 3) for x in self.kmeans.cluster_centers_[cluster_id]]
            }
        
        comparativa_clusters = []
        for cluster_name, cluster_data in cluster_analysis.items():
            comparativa_clusters.append({
                'cluster_id': cluster_data['id'],
                'cluster_nombre': cluster_name,
                'cantidad_entregas': cluster_data['cantidad_entregas'],
                'porcentaje': cluster_data['porcentaje_total'],
                'promedio_general': cluster_data['estadisticas_generales']['promedio_general'],
                'desviacion_general': cluster_data['estadisticas_generales']['desviacion_general']
            })
        
        comparativa_clusters.sort(key=lambda x: x['promedio_general'], reverse=True)
        
        estadisticas_globales = {
            'promedio_todas_entregas': round(float(np.mean(X)), 2),
            'desviacion_todas_entregas': round(float(np.std(X)), 2),
            'valor_minimo_global': round(float(np.min(X)), 2),
            'valor_maximo_global': round(float(np.max(X)), 2),
            'mediana_global': round(float(np.median(X)), 2)
        }
        
        distribucion_por_pregunta = []
        for i, pregunta in enumerate(self.feature_names):
            valores_pregunta = X[:, i]
            opciones_info = pregunta_opciones.get(pregunta, [])
            
            distribucion_por_pregunta.append({
                'pregunta': pregunta,
                'categoria': categorias_por_pregunta[i],
                'promedio': round(float(np.mean(valores_pregunta)), 2),
                'desviacion': round(float(np.std(valores_pregunta)), 2),
                'minimo': round(float(np.min(valores_pregunta)), 2),
                'maximo': round(float(np.max(valores_pregunta)), 2),
                'mediana': round(float(np.median(valores_pregunta)), 2),
                'opciones_disponibles': opciones_info
            })
        
        result = {
            "success": True,
            "encuesta": {
                "nombre": encuesta.get('nombre', 'Sin nombre'),
                "descripcion": encuesta.get('descripcion', ''),
                "categoria_detectada": categoria_encuesta
            },
            "total_entregas": n_entregas,
            "total_preguntas": len(self.feature_names),
            "estadisticas_globales": estadisticas_globales,
            "distribucion_por_pregunta": distribucion_por_pregunta,
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
            "comparativa_clusters": comparativa_clusters,
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
    
    def _interpretar_valor_con_opciones(self, valor: float, opciones: List[Dict]) -> str:
        if not opciones:
            return f"Valor: {round(valor, 2)}"
        
        if valor == 0:
            return "Sin respuesta"
        
        opciones_ordenadas = sorted(opciones, key=lambda x: float(x.get('valor', 0)))
        
        # Buscar coincidencia exacta o la más cercana
        mejor_coincidencia = None
        menor_diferencia = float('inf')
        
        for opcion in opciones_ordenadas:
            try:
                valor_opcion = float(opcion.get('valor', 0))
                diferencia = abs(valor - valor_opcion)
                if diferencia < menor_diferencia:
                    menor_diferencia = diferencia
                    mejor_coincidencia = opcion
            except (ValueError, TypeError):
                continue
        
        if mejor_coincidencia:
            texto = mejor_coincidencia.get('texto', '')
            valor_opcion = mejor_coincidencia.get('valor', '')
            
            # Si la diferencia es muy pequeña (menos de 0.3), considerarla una coincidencia exacta
            if menor_diferencia < 0.3:
                return f"{texto} (valor: {valor_opcion})"
            else:
                # Si la diferencia es grande, mostrar ambos valores
                return f"Cerca de: {texto} (esperado: {valor_opcion}, actual: {round(valor, 2)})"
        
        return f"Valor: {round(valor, 2)}"
    
    def reset_model(self):
        self.is_trained = False
        self.kmeans = None
        self.feature_names = []
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
