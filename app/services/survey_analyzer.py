import json
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class SurveyAnalyzer:
    def __init__(self, data_source):
        """
        Inicializa el analizador de encuestas con clustering K-means
        data_source: puede ser ruta de archivo o datos JSON directos
        """
        self.data_source = data_source
        self.raw_data = None
        self.processed_data = None
        self.feature_matrix = None
        self.scaler = StandardScaler()
        self.kmeans_models = {}
        self.clusters = {}
        self.label_encoders = {}
        
    def load_data(self):
        """Cargar y parsear datos desde archivo o JSON directo"""
        
        try:
            if isinstance(self.data_source, str):
                with open(self.data_source, 'r', encoding='utf-8') as file:
                    self.raw_data = json.load(file)
            elif isinstance(self.data_source, (dict, list)):
                self.raw_data = self.data_source
            else:
                raise ValueError("data_source debe ser una ruta de archivo o datos JSON")
            
            if isinstance(self.raw_data, list):
                print(f"Datos cargados: {len(self.raw_data)} encuestas")
            elif isinstance(self.raw_data, dict) and 'usuarios' in self.raw_data:
                print(f"Datos cargados: {len(self.raw_data['usuarios'])} usuarios")
            else:
                raise ValueError("Formato de datos no reconocido")
                
        except Exception as e:
            raise Exception(f"Error cargando datos: {str(e)}")
        
    def extract_features(self):
        """
        Extrae características de las respuestas para crear matriz de features
        """
        print(" Extrayendo características...")
        
        if isinstance(self.raw_data, list):
            users_features = self._extract_from_survey_list()
        elif isinstance(self.raw_data, dict) and 'usuarios' in self.raw_data:
            users_features = self._extract_from_user_dict()
        else:
            raise ValueError("Formato de datos no reconocido")
        
        self.processed_data = pd.DataFrame(users_features)
        print(f"Características extraídas para {len(users_features)} entregas/usuarios")
    
    def _extract_from_survey_list(self):
        """Extrae características del formato nuevo (lista de encuestas) - CON VALIDACIONES"""
        users_features = []
        
        for survey_data in self.raw_data:
            encuesta_info = survey_data.get('encuesta', {})
            respuestas = survey_data.get('respuestas_usuario', [])
            
            if not respuestas:
                print(f"WARNING: Encuesta '{encuesta_info.get('nombre', 'Unknown')}' sin respuestas, omitiendo...")
                continue
            
            entregas = defaultdict(list)
            for resp in respuestas:
                entrega_id = resp.get('entrega_id')
                if entrega_id:
                    entregas[entrega_id].append(resp)
            
            if not entregas:
                print(f"WARNING: Encuesta '{encuesta_info.get('nombre', 'Unknown')}' sin entregas válidas, omitiendo...")
                continue
            
            for entrega_id, respuestas_entrega in entregas.items():
                textos_analisis = [encuesta_info.get('nombre', ''), encuesta_info.get('descripcion', '')]
                for resp in respuestas_entrega:
                    textos_analisis.append(resp.get('pregunta_texto', ''))
                    textos_analisis.append(str(resp.get('respuesta_seleccionada', '')))
                
                categorias_detectadas = self._detectar_categorias_tematicas(textos_analisis)
                
                user_features = {
                    'usuario_id': entrega_id,
                    'nombre': entrega_id,
                    'encuesta_nombre': encuesta_info.get('nombre', 'Unknown'),
                    'encuesta_id': encuesta_info.get('id', 'Unknown'),
                    'canal': encuesta_info.get('canal', 'Unknown'),
                    'total_encuestas': 1,
                    'total_respuestas': len(respuestas_entrega),
                    'tipos_respuesta': defaultdict(int),
                    'canales': defaultdict(int),
                    'temas_encuesta': defaultdict(int),
                    'categorias_tematicas': categorias_detectadas,
                    'patrones_respuesta': defaultdict(int),
                    'diversidad_respuestas': 0,
                    'score_promedio': 0,
                    'respuestas_problematicas': 0,
                    'respuestas_satisfactorias': 0
                }
                
                user_features['canales'][encuesta_info.get('canal', 'Unknown')] = 1
                
                tema = self._extract_survey_theme(encuesta_info.get('nombre', ''))
                user_features['temas_encuesta'][tema] = 1
                
                valores_numericos = []
                for respuesta in respuestas_entrega:
                    user_features['tipos_respuesta'][respuesta.get('tipo_pregunta', 'Unknown')] += 1
                    
                    valores = respuesta.get('valores_seleccionados', [])
                    
                    valores_respuesta = []
                    for valor in valores:
                        try:
                            val_num = int(valor)
                            valores_numericos.append(val_num)
                            valores_respuesta.append(val_num)
                            user_features['patrones_respuesta'][f'valor_{val_num}'] += 1
                        except (ValueError, TypeError):
                            pass
                    
                    if valores_respuesta:
                        promedio_respuesta = np.mean(valores_respuesta)
                        if promedio_respuesta <= 2.0:
                            user_features['respuestas_problematicas'] += 1
                        elif promedio_respuesta >= 4.0:
                            user_features['respuestas_satisfactorias'] += 1
                
                if valores_numericos:
                    user_features['score_promedio'] = np.mean(valores_numericos)
                    user_features['diversidad_respuestas'] = len(set(valores_numericos))
                
                users_features.append(user_features)
        
        return users_features
    
    def _extract_from_user_dict(self):
        """Extrae características del formato antiguo (dict con usuarios)"""
        users_features = []
        
        for user in self.raw_data['usuarios']:
            user_features = {
                'usuario_id': user['usuario_id'],
                'nombre': user['nombre'],
                'es_admin': int(user['es_admin']),
                'total_encuestas': len(user['encuestas']),
                'total_respuestas': 0,
                'tipos_respuesta': defaultdict(int),
                'canales': defaultdict(int),
                'temas_encuesta': defaultdict(int),
                'patrones_respuesta': defaultdict(int),
                'diversidad_respuestas': 0,
                'tiempo_promedio_respuesta': 0,
                'score_promedio': 0,
                'respuestas_problematicas': 0,
                'respuestas_satisfactorias': 0
            }
            
            all_responses = []
            valores_numericos = []
            
            for encuesta in user['encuestas']:
                user_features['canales'][encuesta['canal']] += 1
                
                tema = self._extract_survey_theme(encuesta['encuesta_nombre'])
                user_features['temas_encuesta'][tema] += 1
                
                for respuesta in encuesta['respuestas']:
                    all_responses.append(respuesta)
                    user_features['total_respuestas'] += 1
                    user_features['tipos_respuesta'][respuesta['tipo_pregunta']] += 1
                    
                    if respuesta['respuesta_valor']:
                        try:
                            valor = int(respuesta['respuesta_valor'])
                            valores_numericos.append(valor)
                            user_features['patrones_respuesta'][f'valor_{valor}'] += 1
                            
                            if valor <= 2:
                                user_features['respuestas_problematicas'] += 1
                            elif valor >= 4:
                                user_features['respuestas_satisfactorias'] += 1
                        except (ValueError, TypeError):
                            pass
            
            if valores_numericos:
                user_features['score_promedio'] = np.mean(valores_numericos)
                user_features['diversidad_respuestas'] = len(set(valores_numericos))
            
            users_features.append(user_features)
        
        return users_features
        
    def _extract_survey_theme(self, survey_name):
        """Extrae el tema principal de la encuesta"""
        survey_name = survey_name.lower()
        
        if any(word in survey_name for word in ['robot', 'pick', 'place', 'precision', 'altura']):
            return 'robotica'
        elif any(word in survey_name for word in ['seguridad', 'emergencia', 'cortina', 'laser']):
            return 'seguridad'
        elif any(word in survey_name for word in ['mantenimiento', 'predictivo', 'telemetria', 'torque']):
            return 'mantenimiento'
        elif any(word in survey_name for word in ['wifi', 'voip', 'ticket', 'red']):
            return 'it_infraestructura'
        elif any(word in survey_name for word in ['api', 'desarrollo', 'sso', 'sistema']):
            return 'desarrollo'
        elif any(word in survey_name for word in ['noc', 'monitoreo', 'continuidad']):
            return 'monitoreo'
        elif any(word in survey_name for word in ['madera', 'forestal', 'corte', 'secado']):
            return 'forestal'
        else:
            return 'otros'
    
    def _detectar_categorias_tematicas(self, textos):
        """
        Detecta categorías temáticas específicas basadas en palabras clave del dominio
        """
        categorias = {
            "rendimiento": [
                "precision", "velocidad", "ciclo", "cpm", "eficiencia", "estabilidad",
                "disponibilidad", "consumo", "productividad", "repetibilidad", "fragmentacion",
                "recuperacion", "caudal", "presion"
            ],
            "calidad_conformidad": [
                "consistencia", "tolerancia", "conformidad", "defectos", 
                "control_calidad", "satisfaccion", "pureza", "uniformidad",
                "solidez", "grosor", "impurezas"
            ],
            "seguridad_cumplimiento": [
                "seguridad", "riesgo", "incidente", "parada_emergencia", "resguardo", 
                "norma", "certificacion", "inspeccion", "trazabilidad",
                "vibracion", "polvo", "epp"
            ],
            "operaciones_mantenimiento": [
                "configuracion", "ajuste", "parametro", "mantenimiento", "diagnostico",
                "telemetria", "monitoreo", "alerta", "scada", "muestreo",
                "repuestos", "ruido", "garantia"
            ],
            "experiencia_usuario": [
                "comunicacion", "facilidad", "puntualidad", "soporte",
                "capacitacion", "recomendacion", "satisfaccion", "documentacion",
                "atencion", "rapidez"
            ],
            "robotica_automatizacion": [
                "robot", "pick", "place", "celda", "end-effector", "lidar",
                "calibracion", "altura", "actuador", "sensor", "trayectoria"
            ]
        }
        
        texto_completo = ' '.join([str(t).lower() for t in textos if t])
        
        categorias_detectadas = {}
        for categoria, keywords in categorias.items():
            matches = sum(1 for keyword in keywords if keyword in texto_completo)
            if matches > 0:
                categorias_detectadas[categoria] = matches
        
        return categorias_detectadas
    
    def create_feature_matrix(self):
        """
        Crea matriz de características numéricas para K-means
        """
        print("Creando matriz de características...")
        
        numeric_features = [
            'total_encuestas', 'total_respuestas', 
            'diversidad_respuestas', 'score_promedio',
            'respuestas_problematicas', 'respuestas_satisfactorias'
        ]
        
        for col in numeric_features:
            if col not in self.processed_data.columns:
                self.processed_data[col] = 0
        
        feature_matrix = self.processed_data[numeric_features].copy()
        
        for tipo in ['Opción Múltiple', 'Opción Única']:
            feature_matrix[f'tipo_{tipo.replace(" ", "_").lower()}'] = [
                user['tipos_respuesta'].get(tipo, 0) for user in self.processed_data.to_dict('records')
            ]
        
        all_canales = set()
        for user in self.processed_data.to_dict('records'):
            all_canales.update(user['canales'].keys())
        
        for canal in all_canales:
            feature_matrix[f'canal_{canal.lower()}'] = [
                user['canales'].get(canal, 0) for user in self.processed_data.to_dict('records')
            ]
        
        all_temas = set()
        for user in self.processed_data.to_dict('records'):
            all_temas.update(user['temas_encuesta'].keys())
        
        for tema in all_temas:
            feature_matrix[f'tema_{tema}'] = [
                user['temas_encuesta'].get(tema, 0) for user in self.processed_data.to_dict('records')
            ]
        
        all_patterns = Counter()
        for user in self.processed_data.to_dict('records'):
            all_patterns.update(user['patrones_respuesta'].keys())
        
        top_patterns = [pattern for pattern, _ in all_patterns.most_common(10)]
        
        for pattern in top_patterns:
            feature_matrix[f'patron_{pattern}'] = [
                user['patrones_respuesta'].get(pattern, 0) for user in self.processed_data.to_dict('records')
            ]
        
        self.feature_matrix = feature_matrix
        print(f"Matriz creada: {feature_matrix.shape[0]} usuarios x {feature_matrix.shape[1]} características")
        
    def find_optimal_clusters(self, max_k=6):
        """
        Encuentra el número óptimo de clusters usando método del codo y silhouette
        Recomendado: 4-6 clusters para mejor balance entre detalle y manejabilidad
        """
        print(" Buscando número óptimo de clusters...")
        
        X_scaled = self.scaler.fit_transform(self.feature_matrix)
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, len(self.feature_matrix)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, clusters))
        
        self._plot_cluster_metrics(k_range, inertias, silhouette_scores)
        
        best_k = k_range[np.argmax(silhouette_scores)]
        
        if best_k > 6:
            print(f"WARNING: Silhouette sugiere {best_k} clusters, pero se recomienda 4-6 para mejor manejabilidad")
            scores_in_range = [(k, silhouette_scores[k-2]) for k in range(4, min(7, len(k_range)+2)) if k in k_range]
            if scores_in_range:
                best_k = max(scores_in_range, key=lambda x: x[1])[0]
                print(f"Usando k={best_k} (balance óptimo entre precisión y manejabilidad)")
        else:
            print(f"Número óptimo de clusters: {best_k}")
        
        return best_k, silhouette_scores[best_k-2] if best_k-2 < len(silhouette_scores) else max(silhouette_scores)
    
    def _plot_cluster_metrics(self, k_range, inertias, silhouette_scores):
        """Grafica métricas de clustering"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(k_range, inertias, 'bo-')
        ax1.set_xlabel('Número de Clusters (k)')
        ax1.set_ylabel('Inercia')
        ax1.set_title('Método del Codo')
        ax1.grid(True)
        
        ax2.plot(k_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Número de Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Análisis Silhouette')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def perform_clustering(self, n_clusters=None):
        """
        Realiza clustering K-means
        """
        if n_clusters is None:
            n_clusters, _ = self.find_optimal_clusters()
        
        print(f" Realizando clustering con {n_clusters} clusters...")
        
        X_scaled = self.scaler.fit_transform(self.feature_matrix)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        self.kmeans_models['main'] = kmeans
        self.clusters['main'] = cluster_labels
        
        self.processed_data['cluster'] = cluster_labels
        
        print(f"Clustering completado!")
        
        return cluster_labels
    
    def analyze_clusters(self):
        """
        Analiza y describe los clusters encontrados con detalle descriptivo - CON VALIDACIONES
        """
        print(" Analizando clusters con detalle...")
        
        if 'main' not in self.clusters or len(self.clusters['main']) == 0:
            raise ValueError("No se han generado clusters. Ejecuta perform_clustering() primero.")
        
        cluster_analysis = {}
        
        for cluster_id in range(len(np.unique(self.clusters['main']))):
            cluster_data = self.processed_data[self.processed_data['cluster'] == cluster_id]
            
            if len(cluster_data) == 0:
                print(f"WARNING: Cluster {cluster_id} está vacío, omitiendo...")
                continue
            
            try:
                score_promedio = float(cluster_data['score_promedio'].mean()) if 'score_promedio' in cluster_data.columns else 0
                if np.isnan(score_promedio) or np.isinf(score_promedio):
                    print(f"WARNING: Cluster {cluster_id} tiene score inválido, usando 0.0")
                    score_promedio = 0.0
            except Exception as e:
                print(f"WARNING: Error calculando score para Cluster {cluster_id}: {e}")
                score_promedio = 0.0
            
            total_usuarios = len(cluster_data)
            
            entregas_criticas = len(cluster_data[cluster_data['score_promedio'] <= 2.0])
            entregas_satisfactorias = len(cluster_data[cluster_data['score_promedio'] >= 4.0])
            entregas_regulares = total_usuarios - entregas_criticas - entregas_satisfactorias
            
            porcentaje_entregas_criticas = (entregas_criticas / total_usuarios * 100) if total_usuarios > 0 else 0
            porcentaje_entregas_satisfactorias = (entregas_satisfactorias / total_usuarios * 100) if total_usuarios > 0 else 0
            porcentaje_entregas_regulares = (entregas_regulares / total_usuarios * 100) if total_usuarios > 0 else 0
            
            total_resp = float(cluster_data['total_respuestas'].sum())
            resp_problematicas = float(cluster_data['respuestas_problematicas'].sum()) if 'respuestas_problematicas' in cluster_data.columns else 0
            resp_satisfactorias = float(cluster_data['respuestas_satisfactorias'].sum()) if 'respuestas_satisfactorias' in cluster_data.columns else 0
            resp_regulares = total_resp - resp_problematicas - resp_satisfactorias
            
            porcentaje_resp_problematicas = (resp_problematicas / total_resp * 100) if total_resp > 0 else 0
            porcentaje_resp_satisfactorias = (resp_satisfactorias / total_resp * 100) if total_resp > 0 else 0
            porcentaje_resp_regulares = (resp_regulares / total_resp * 100) if total_resp > 0 else 0
            
            if np.isnan(score_promedio):
                score_promedio = 0.0
            
            categoria = self._categorizar_cluster_extendido(score_promedio)
            nombres_cluster = self._obtener_nombre_cluster(categoria)
            
            descripciones = {
                'critico': f"Este grupo presenta un rendimiento crítico (score: {score_promedio:.2f}/5.0). Los equipos o sistemas evaluados muestran fallas graves que requieren intervención inmediata.",
                'riesgo': f"Este grupo muestra un rendimiento bajo el estándar (score: {score_promedio:.2f}/5.0). Existen problemas significativos que impactan la operación y requieren atención prioritaria.",
                'atencion': f"Este grupo muestra un desempeño aceptable pero mejorable (score: {score_promedio:.2f}/5.0). Existen oportunidades de optimización para alcanzar niveles superiores de rendimiento.",
                'satisfactorio': f"Este grupo alcanza un buen nivel de desempeño (score: {score_promedio:.2f}/5.0). Los sistemas funcionan correctamente con ligeras áreas de mejora.",
                'excelente': f"Este grupo representa el estándar de excelencia (score: {score_promedio:.2f}/5.0). Los sistemas operan de manera óptima y pueden servir como referencia para otros grupos."
            }
            
            estados = {
                'critico': 'Funcionamiento Crítico',
                'riesgo': 'Funcionamiento Deficiente',
                'atencion': 'Funcionamiento Regular',
                'satisfactorio': 'Funcionamiento Bueno',
                'excelente': 'Funcionamiento Óptimo'
            }
            
            urgencias = {
                'critico': 'EMERGENCIA',
                'riesgo': 'ALTA',
                'atencion': 'MEDIA',
                'satisfactorio': 'BAJA',
                'excelente': 'NINGUNA'
            }
            
            tipo = nombres_cluster['nombre_completo']
            descripcion = descripciones[categoria]
            estado_general = estados[categoria]
            nivel_urgencia = urgencias[categoria]
            
            analysis = {
                'cluster_id': f'Cluster_{cluster_id}',
                'nombre_cluster': nombres_cluster['nombre_corto'],
                'nombre_completo': nombres_cluster['nombre_completo'],
                'titulo_ejecutivo': nombres_cluster['titulo_ejecutivo'],
                'subtitulo_ingles': nombres_cluster['subtitulo_ingles'],
                'tipo_cluster': tipo,
                'categoria': categoria,
                'estado_general': estado_general,
                'nivel_urgencia': nivel_urgencia,
                'descripcion_detallada': descripcion,
                'tamano_grupo': {
                    'cantidad': int(len(cluster_data)),
                    'porcentaje_total': round(len(cluster_data) / len(self.processed_data) * 100, 2),
                    'descripcion': f"{len(cluster_data)} entregas de un total de {len(self.processed_data)} ({len(cluster_data) / len(self.processed_data) * 100:.1f}%)"
                },
                'entregas_ids': cluster_data['nombre'].head(10).tolist(),
                'entregas_ejemplo': cluster_data['nombre'].head(3).tolist()
            }
            
            categorias_cluster = defaultdict(int)
            for _, row in cluster_data.iterrows():
                if 'categorias_tematicas' in row and isinstance(row['categorias_tematicas'], dict):
                    for cat, count in row['categorias_tematicas'].items():
                        categorias_cluster[cat] += count
            
            if categorias_cluster:
                analysis['categorias_tematicas_detectadas'] = dict(sorted(
                    categorias_cluster.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5])
            
            analysis['metricas_rendimiento'] = {
                'score_promedio': {
                    'valor': round(score_promedio, 2),
                    'sobre': 5.0,
                    'porcentaje': round(score_promedio / 5.0 * 100, 1),
                    'interpretacion': self._interpretar_score(score_promedio)
                },
                'distribucion_entregas': {
                    'criticas': {
                        'cantidad': int(entregas_criticas),
                        'porcentaje': round(porcentaje_entregas_criticas, 1),
                        'descripcion': f"{int(entregas_criticas)} entregas con score ≤2.0 ({porcentaje_entregas_criticas:.1f}% del cluster)"
                    },
                    'satisfactorias': {
                        'cantidad': int(entregas_satisfactorias),
                        'porcentaje': round(porcentaje_entregas_satisfactorias, 1),
                        'descripcion': f"{int(entregas_satisfactorias)} entregas con score ≥4.0 ({porcentaje_entregas_satisfactorias:.1f}% del cluster)"
                    },
                    'regulares': {
                        'cantidad': int(entregas_regulares),
                        'porcentaje': round(porcentaje_entregas_regulares, 1),
                        'descripcion': f"{int(entregas_regulares)} entregas con score 2.1-3.9 ({porcentaje_entregas_regulares:.1f}% del cluster)"
                    }
                },
                'distribucion_respuestas': {
                    'problematicas': {
                        'cantidad': int(resp_problematicas),
                        'porcentaje': round(porcentaje_resp_problematicas, 1),
                        'descripcion': f"{int(resp_problematicas)} respuestas con valor 1-2 de {int(total_resp)} totales ({porcentaje_resp_problematicas:.1f}%)"
                    },
                    'satisfactorias': {
                        'cantidad': int(resp_satisfactorias),
                        'porcentaje': round(porcentaje_resp_satisfactorias, 1),
                        'descripcion': f"{int(resp_satisfactorias)} respuestas con valor 4-5 de {int(total_resp)} totales ({porcentaje_resp_satisfactorias:.1f}%)"
                    },
                    'regulares': {
                        'cantidad': int(resp_regulares),
                        'porcentaje': round(porcentaje_resp_regulares, 1),
                        'descripcion': f"{int(resp_regulares)} respuestas con valor 3 de {int(total_resp)} totales ({porcentaje_resp_regulares:.1f}%)"
                    }
                },
                'totales': {
                    'entregas': int(total_usuarios),
                    'respuestas': int(total_resp),
                    'promedio_respuestas_por_entrega': round(total_resp / total_usuarios, 1) if total_usuarios > 0 else 0
                }
            }
            
            patron_respuestas = self._analizar_patrones_respuesta(cluster_data)
            analysis['patrones_detectados'] = patron_respuestas
            
            all_temas = defaultdict(int)
            for user_data in cluster_data.to_dict('records'):
                for tema, count in user_data['temas_encuesta'].items():
                    all_temas[tema] += count
            
            temas_ordenados = sorted(all_temas.items(), key=lambda x: x[1], reverse=True)
            analysis['areas_evaluadas'] = {
                'temas_principales': [
                    {
                        'tema': self._traducir_tema(tema),
                        'tema_codigo': tema,
                        'frecuencia': int(count),
                        'descripcion': self._describir_tema(tema)
                    }
                    for tema, count in temas_ordenados[:3]
                ]
            }
            
            analysis['estadisticas'] = {
                'total_encuestas': {
                    'promedio_por_usuario': round(float(cluster_data['total_encuestas'].mean()), 2),
                    'total': int(cluster_data['total_encuestas'].sum())
                },
                'total_respuestas': {
                    'promedio_por_usuario': round(float(cluster_data['total_respuestas'].mean()), 2),
                    'total': int(cluster_data['total_respuestas'].sum())
                },
                'diversidad_respuestas': {
                    'promedio': round(float(cluster_data['diversidad_respuestas'].mean()), 2),
                    'interpretacion': self._interpretar_diversidad(float(cluster_data['diversidad_respuestas'].mean()))
                }
            }
            
            analysis['indicadores_dashboard'] = {
                'prioridad': self._obtener_prioridad_numerica(nivel_urgencia),
                'resumen_ejecutivo': self._generar_resumen_ejecutivo(
                    len(cluster_data), score_promedio, porcentaje_entregas_criticas, temas_ordenados
                )
            }
            
            cluster_analysis[f'Cluster_{cluster_id}'] = analysis
        
        self.cluster_analysis = cluster_analysis
        return cluster_analysis
    
    def _interpretar_score(self, score):
        """Interpreta el score numérico en lenguaje descriptivo"""
        if score <= 1.5:
            return "Rendimiento crítico - Los sistemas presentan fallas graves que requieren atención inmediata"
        elif score <= 2.5:
            return "Rendimiento bajo - Existen problemas significativos que impactan la operación"
        elif score <= 3.5:
            return "Rendimiento aceptable - Funcionamiento estándar con oportunidades de mejora"
        elif score <= 4.5:
            return "Buen rendimiento - Los sistemas operan correctamente con alta confiabilidad"
        else:
            return "Rendimiento excelente - Operación óptima que supera las expectativas"
    
    def _analizar_patrones_respuesta(self, cluster_data):
        """Analiza patrones específicos en las respuestas"""
        patrones = []
        
        for user_data in cluster_data.to_dict('records'):
            patron_resp = user_data.get('patrones_respuesta', {})
            for patron, count in patron_resp.items():
                if count > 0:
                    valor = patron.replace('valor_', '')
                    patrones.append({
                        'patron': patron,
                        'valor_numerico': valor,
                        'frecuencia': count,
                        'interpretacion': self._interpretar_valor_respuesta(valor)
                    })
        
        patrones_consolidados = defaultdict(lambda: {'frecuencia': 0, 'interpretacion': ''})
        for p in patrones:
            key = p['valor_numerico']
            patrones_consolidados[key]['frecuencia'] += p['frecuencia']
            patrones_consolidados[key]['interpretacion'] = p['interpretacion']
        
        return [
            {
                'valor': k,
                'frecuencia': v['frecuencia'],
                'interpretacion': v['interpretacion']
            }
            for k, v in sorted(patrones_consolidados.items(), key=lambda x: x[1]['frecuencia'], reverse=True)[:5]
        ]
    
    def _interpretar_valor_respuesta(self, valor):
        """Interpreta el significado de cada valor de respuesta"""
        try:
            val = int(valor)
            interpretaciones = {
                1: "Muy insatisfecho / No funciona / Crítico",
                2: "Insatisfecho / Funciona mal / Problemático",
                3: "Neutral / Funciona aceptablemente / Regular",
                4: "Satisfecho / Funciona bien / Bueno",
                5: "Muy satisfecho / Funciona excelente / Óptimo"
            }
            return interpretaciones.get(val, "Valor no estándar")
        except:
            return "Respuesta no numérica"
    
    def _traducir_tema(self, tema_codigo):
        """Traduce el código del tema a nombre legible"""
        traducciones = {
            'robotica': 'Robótica y Automatización',
            'seguridad': 'Seguridad Industrial',
            'mantenimiento': 'Mantenimiento Predictivo',
            'it_infraestructura': 'Infraestructura IT',
            'desarrollo': 'Desarrollo de Sistemas',
            'monitoreo': 'Monitoreo y NOC',
            'forestal': 'Industria Forestal',
            'otros': 'Otros'
        }
        return traducciones.get(tema_codigo, tema_codigo.title())
    
    def _describir_tema(self, tema_codigo):
        """Proporciona descripción detallada del tema"""
        descripciones = {
            'robotica': 'Evaluación de precisión, rendimiento y fiabilidad de sistemas robóticos y celdas automatizadas',
            'seguridad': 'Análisis de sistemas de seguridad funcional, resguardos y paradas de emergencia',
            'mantenimiento': 'Monitoreo de telemetría, mantenimiento predictivo y análisis de fallas',
            'it_infraestructura': 'Evaluación de redes, Wi-Fi, VoIP y sistemas de soporte IT',
            'desarrollo': 'Análisis de APIs, integraciones y desarrollo de sistemas',
            'monitoreo': 'Sistemas de monitoreo NOC, alertas y continuidad operacional',
            'forestal': 'Evaluación de procesos forestales, corte, secado y clasificación',
            'otros': 'Otras áreas de evaluación'
        }
        return descripciones.get(tema_codigo, 'Área de evaluación no categorizada')
    
    def _interpretar_diversidad(self, diversidad):
        """Interpreta la diversidad de respuestas"""
        if diversidad <= 1:
            return "Respuestas muy uniformes - Todos respondieron de forma similar"
        elif diversidad <= 2:
            return "Poca variación - Respuestas concentradas en pocas opciones"
        elif diversidad <= 3:
            return "Variación moderada - Mix de diferentes tipos de respuestas"
        else:
            return "Alta variación - Respuestas muy diversas entre los usuarios"
    
    def _generar_recomendaciones_detalladas(self, categoria, score, porcentaje_entregas_criticas, temas):
        """Genera recomendaciones específicas y accionables -  CORREGIDAS para evitar contradicciones"""
        recomendaciones = {
            'acciones_inmediatas': [],
            'mejoras_corto_plazo': [],
            'estrategias_largo_plazo': []
        }
        
        if categoria == "critico":
            recomendaciones['acciones_inmediatas'] = [
                " URGENTE: Revisar inmediatamente todos los equipos/sistemas con fallas",
                " Contactar usuarios afectados en menos de 24 horas",
                " Asignar equipo técnico especializado para diagnóstico urgente",
                " Crear plan de acción correctivo inmediato con métricas claras"
            ]
            recomendaciones['mejoras_corto_plazo'] = [
                "Implementar monitoreo 24/7 de los sistemas críticos",
                "Establecer protocolo de escalamiento para incidentes similares",
                "Auditoría completa de procesos y procedimientos"
            ]
            
        elif categoria == "riesgo":
            recomendaciones['acciones_inmediatas'] = [
                " Programar revisión técnica dentro de 48-72 horas",
                " Contactar usuarios para entender problemas específicos",
                " Analizar logs y métricas para identificar causas raíz",
                " Preparar plan de mejora con timeline definido"
            ]
            recomendaciones['mejoras_corto_plazo'] = [
                "Implementar monitoreo preventivo de sistemas en riesgo",
                "Revisar y actualizar procedimientos operativos",
                "Capacitación específica al personal involucrado"
            ]
            
        elif categoria == "atencion":
            recomendaciones['acciones_inmediatas'] = [
                " Programar revisión técnica preventiva (1-2 semanas)",
                " Analizar métricas para identificar oportunidades de mejora",
                " Reunir feedback de usuarios para entender expectativas"
            ]
            recomendaciones['mejoras_corto_plazo'] = [
                "Optimizar configuraciones y parámetros operativos",
                "Implementar mejoras incrementales basadas en datos",
                "Establecer KPIs y benchmarks objetivo"
            ]
            
        elif categoria == "satisfactorio":
            recomendaciones['acciones_inmediatas'] = [
                " Documentar configuraciones y procedimientos exitosos",
                " Monitorear continuidad del buen desempeño",
                " Identificar factores clave del éxito"
            ]
            recomendaciones['mejoras_corto_plazo'] = [
                "Mantener nivel actual con monitoreo periódico",
                "Buscar oportunidades de optimización incremental",
                "Compartir mejores prácticas con otros equipos"
            ]
            
        elif categoria == "excelente":
            recomendaciones['acciones_inmediatas'] = [
                " Documentar como caso de éxito y referencia",
                " Crear programa de mentoría basado en este grupo",
                " Analizar factores diferenciadores para replicar"
            ]
            recomendaciones['mejoras_corto_plazo'] = [
                "Establecer como benchmark para otros clusters",
                "Explorar innovaciones para mantener liderazgo",
                "Reconocer y celebrar el desempeño excepcional"
            ]
        
        if temas:
            tema_principal = temas[0][0]
            recomendaciones['especificas_por_area'] = self._recomendaciones_por_tema(tema_principal, categoria)
        else:
            recomendaciones['especificas_por_area'] = ["Seguir mejores prácticas del sector"]
        
        recomendaciones['estrategias_largo_plazo'] = [
            "Implementar sistema de mejora continua basado en datos",
            "Establecer programa de capacitación continua",
            "Desarrollar cultura de excelencia operacional",
            "Crear ciclos de feedback y aprendizaje organizacional"
        ]
        
        return recomendaciones
    
    def _recomendaciones_por_tema(self, tema, categoria):
        """Recomendaciones específicas según el tema"""
        recomendaciones_tema = {
            'robotica': {
                'critico': ["Recalibrar sistemas robóticos", "Revisar sensores y actuadores", "Verificar programación de trayectorias"],
                'atencion': ["Optimizar ciclos de trabajo", "Ajustar parámetros de precisión", "Actualizar algoritmos de control"],
                'satisfactorio': ["Mantener calendarios de calibración", "Monitorear desgaste de componentes"]
            },
            'seguridad': {
                'critico': ["Auditoría completa de sistemas de seguridad", "Verificar todos los enclavamientos", "Probar paradas de emergencia"],
                'atencion': ["Revisar configuración de zonas de seguridad", "Actualizar protocolos de respuesta", "Capacitar en procedimientos"],
                'satisfactorio': ["Mantener certificaciones vigentes", "Realizar simulacros periódicos"]
            },
            'mantenimiento': {
                'critico': ["Implementar mantenimiento correctivo urgente", "Analizar causas raíz de fallas", "Renovar componentes críticos"],
                'atencion': ["Ajustar frecuencia de mantenimiento preventivo", "Implementar monitoreo predictivo", "Actualizar procedimientos"],
                'satisfactorio': ["Continuar con plan de mantenimiento actual", "Evaluar nuevas tecnologías predictivas"]
            }
        }
        
        return recomendaciones_tema.get(tema, {}).get(categoria, ["Seguir mejores prácticas del sector"])
    
    def _obtener_nombre_cluster(self, categoria):
        """
        Genera nombres descriptivos y ejecutivos para cada cluster
        Sistema de 5 niveles para mejor manejabilidad
        """
        nombres = {
            'excelente': {
                'nivel': 'EXCELENCIA',
                'titulo': 'Referentes de Alto Desempeño',
                'subtitulo': 'Best Performers',
                'descripcion': 'Sobresaliente'
            },
            'satisfactorio': {
                'nivel': 'ÓPTIMO',
                'titulo': 'Operación en Estándar',
                'subtitulo': 'Standard Compliance',
                'descripcion': 'Buen Rendimiento'
            },
            'atencion': {
                'nivel': 'MEJORA',
                'titulo': 'Atención Preventiva',
                'subtitulo': 'Early Warning',
                'descripcion': 'Necesita Mejoras'
            },
            'riesgo': {
                'nivel': 'RIESGO',
                'titulo': 'Intervención Necesaria',
                'subtitulo': 'Action Required',
                'descripcion': 'Bajo Desempeño'
            },
            'critico': {
                'nivel': 'CRÍTICO',
                'titulo': 'Emergencia Operacional',
                'subtitulo': 'Critical Priority',
                'descripcion': 'Requiere Intervención Inmediata'
            }
        }
        
        info = nombres.get(categoria, nombres['atencion'])
        
        return {
            'nombre_corto': info['nivel'],
            'nombre_completo': f"{info['nivel']} - {info['descripcion']}",
            'titulo_ejecutivo': info['titulo'],
            'subtitulo_ingles': info['subtitulo']
        }
    
    def _categorizar_cluster_extendido(self, score_promedio):
        """
        Categorización extendida en 5 niveles para mejor diferenciación
        
        Rangos:
        - Excelente: 4.0-5.0 (80-100%)
        - Satisfactorio: 3.5-3.9 (70-79%)
        - Atención: 2.5-3.4 (50-69%)
        - Riesgo: 2.0-2.4 (40-49%)
        - Crítico: 1.0-1.9 (20-39%)
        """
        if score_promedio >= 4.0:
            return 'excelente'
        elif score_promedio >= 3.5:
            return 'satisfactorio'
        elif score_promedio >= 2.5:
            return 'atencion'
        elif score_promedio >= 2.0:
            return 'riesgo'
        else:
            return 'critico'
    
    def _obtener_prioridad_numerica(self, nivel_urgencia):
        """Convierte nivel de urgencia a número para ordenamiento"""
        prioridades = {
            'ALTA': 1,
            'MEDIA': 2,
            'BAJA': 3,
            'NINGUNA': 4
        }
        return prioridades.get(nivel_urgencia, 5)
    
    def _generar_resumen_ejecutivo(self, cantidad, score, porcentaje_entregas_criticas, temas):
        """Genera resumen ejecutivo de una línea para dashboard - CORREGIDO"""
        tema_principal = self._traducir_tema(temas[0][0]) if temas else "Múltiples áreas"
        
        if score <= 2.0:
            nivel_severidad = "crítico" if porcentaje_entregas_criticas > 50 else "serio"
            return f"{cantidad} entregas con rendimiento {nivel_severidad} en {tema_principal} - Requiere atención inmediata"
        elif score <= 3.0:
            return f"{cantidad} entregas con rendimiento regular en {tema_principal} - Oportunidades de mejora identificadas"
        elif score <= 4.0:
            return f"{cantidad} entregas con buen desempeño en {tema_principal} - Operación satisfactoria"
        else:
            return f"{cantidad} entregas con excelente rendimiento en {tema_principal} - Referencia de mejores prácticas"
    
    def visualize_clusters(self):
        """
        Visualiza los clusters usando PCA
        """
        print(" Creando visualizaciones...")
        
        X_scaled = self.scaler.transform(self.feature_matrix)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(self.clusters['main']))))
        
        for i, cluster_id in enumerate(np.unique(self.clusters['main'])):
            cluster_points = X_pca[self.clusters['main'] == cluster_id]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=[colors[i]], label=f'Cluster {cluster_id}', 
                       alpha=0.7, s=50)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
        plt.title('Visualización de Clusters (PCA)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('clusters_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self._plot_cluster_heatmap()
    
    def _plot_cluster_heatmap(self):
        """Crea heatmap de características por cluster"""
        cluster_features = []
        
        for cluster_id in range(len(np.unique(self.clusters['main']))):
            cluster_data = self.processed_data[self.processed_data['cluster'] == cluster_id]
            features = self.feature_matrix[self.processed_data['cluster'] == cluster_id].mean()
            cluster_features.append(features.values)
        
        cluster_features = np.array(cluster_features)
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(cluster_features, 
                   xticklabels=self.feature_matrix.columns,
                   yticklabels=[f'Cluster {i}' for i in range(len(cluster_features))],
                   annot=True, fmt='.2f', cmap='viridis')
        
        plt.title('Heatmap de Características por Cluster')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('cluster_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_cluster(self, user_features):
        """
        Predice el cluster para nuevos datos de usuario
        """
        user_features_scaled = self.scaler.transform([user_features])
        
        cluster = self.kmeans_models['main'].predict(user_features_scaled)[0]
        
        return cluster
    
    def generate_report(self):
        """
        Genera reporte completo del análisis
        """
        print("\n" + "="*50)
        print("REPORTE DE ANÁLISIS DE CLUSTERS")
        print("="*50)
        
        print(f"\n RESUMEN GENERAL:")
        print(f"- Total de usuarios analizados: {len(self.processed_data)}")
        print(f"- Número de características: {self.feature_matrix.shape[1]}")
        print(f"- Número de clusters encontrados: {len(np.unique(self.clusters['main']))}")
        
        print(f"\n ANÁLISIS POR CLUSTER:")
        for cluster_name, analysis in self.cluster_analysis.items():
            print(f"\n--- {cluster_name} ---")
            print(f"Tamaño: {analysis['size']} usuarios ({analysis['percentage']:.1f}%)")
            print(f"Usuarios típicos: {', '.join(analysis['typical_users'])}")
            print(f"Temas principales: {list(analysis['main_themes'].keys())}")
            print(f"Promedio encuestas: {analysis['characteristics']['total_encuestas']['mean']:.1f}")
            print(f"Promedio respuestas: {analysis['characteristics']['total_respuestas']['mean']:.1f}")
        
        print(f"\n INSIGHTS:")
        insights = self._generate_insights()
        for insight in insights:
            print(f"- {insight}")
        
        print("\n" + "="*50)
    
    def _generate_insights(self):
        """Genera insights automáticos del análisis"""
        insights = []
        
        largest_cluster = max(self.cluster_analysis.items(), 
                            key=lambda x: x[1]['size'])
        insights.append(f"El cluster más grande es {largest_cluster[0]} con {largest_cluster[1]['size']} usuarios")
        
        most_active = max(self.cluster_analysis.items(),
                         key=lambda x: x[1]['characteristics']['total_respuestas']['mean'])
        insights.append(f"Los usuarios más activos están en {most_active[0]} (promedio {most_active[1]['characteristics']['total_respuestas']['mean']:.0f} respuestas)")
        
        all_themes = set()
        for analysis in self.cluster_analysis.values():
            all_themes.update(analysis['main_themes'].keys())
        insights.append(f"Se identificaron {len(all_themes)} temas principales: {', '.join(all_themes)}")
        
        return insights
    
    def save_model(self, filepath):
        """Guarda el modelo entrenado"""
        import pickle
        
        model_data = {
            'kmeans_model': self.kmeans_models['main'],
            'scaler': self.scaler,
            'feature_columns': list(self.feature_matrix.columns),
            'cluster_analysis': self.cluster_analysis
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Modelo guardado en: {filepath}")
    
    def export_analysis_json(self, filepath='cluster_analysis_clean.json'):
        """Exporta el análisis a JSON limpio y serializable"""
        if not hasattr(self, 'cluster_analysis') or not self.cluster_analysis:
            raise ValueError("No hay análisis disponible. Ejecuta analyze_clusters() primero.")
        
        def clean_for_json(obj):
            """Limpia objetos para serialización JSON"""
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, defaultdict):
                return dict(obj)
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        clean_analysis = clean_for_json(self.cluster_analysis)
        
        output = {
            "metadata": {
                "fecha_analisis": pd.Timestamp.now().isoformat(),
                "total_entregas": len(self.processed_data),
                "total_clusters": len(clean_analysis),
                "features_utilizadas": list(self.feature_matrix.columns) if hasattr(self, 'feature_matrix') else []
            },
            "cluster_analysis": clean_analysis
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"Análisis exportado a: {filepath}")
        return filepath
    
    def compare_clusters(self):
        """Compara clusters entre sí para identificar diferencias clave"""
        if not hasattr(self, 'cluster_analysis') or not self.cluster_analysis:
            raise ValueError("No hay análisis disponible. Ejecuta analyze_clusters() primero.")
        
        print("\n COMPARACIÓN ENTRE CLUSTERS\n")
        print("="*70)
        
        clusters_by_score = sorted(
            self.cluster_analysis.items(),
            key=lambda x: x[1]['metricas_rendimiento']['score_promedio']['valor'],
            reverse=True
        )
        
        print("\nRANKING POR SCORE:")
        for rank, (cluster_name, data) in enumerate(clusters_by_score, 1):
            score = data['metricas_rendimiento']['score_promedio']['valor']
            categoria = data['categoria']
            tamaño = data['tamano_grupo']['cantidad']
            print(f"{rank}. {cluster_name}: {score:.2f}/5.0 ({categoria}) - {tamaño} entregas")
        
        mejor = clusters_by_score[0]
        peor = clusters_by_score[-1]
        
        print(f"\nMEJOR CLUSTER: {mejor[0]}")
        print(f"   Score: {mejor[1]['metricas_rendimiento']['score_promedio']['valor']:.2f}")
        if mejor[1]['areas_evaluadas']['temas_principales']:
            print(f"   Tema: {mejor[1]['areas_evaluadas']['temas_principales'][0]['tema']}")
        
        print(f"\nCLUSTER CON MÁS OPORTUNIDADES: {peor[0]}")
        print(f"   Score: {peor[1]['metricas_rendimiento']['score_promedio']['valor']:.2f}")
        if peor[1]['areas_evaluadas']['temas_principales']:
            print(f"   Tema: {peor[1]['areas_evaluadas']['temas_principales'][0]['tema']}")
        
        diferencia = mejor[1]['metricas_rendimiento']['score_promedio']['valor'] - \
                     peor[1]['metricas_rendimiento']['score_promedio']['valor']
        
        print(f"\n BRECHA DE RENDIMIENTO: {diferencia:.2f} puntos")
        
        if diferencia > 2.0:
            print("   WARNING: Brecha significativa - Requiere intervención prioritaria")
        elif diferencia > 1.0:
            print("    Brecha moderada - Oportunidad de mejora clara")
        else:
            print("    Brecha pequeña - Rendimiento relativamente homogéneo")
        
        print("\n" + "="*70)
        
        return {
            'ranking': clusters_by_score,
            'mejor_cluster': mejor[0],
            'peor_cluster': peor[0],
            'brecha': diferencia
        }
    
    def get_executive_summary(self):
        """Genera resumen ejecutivo del análisis"""
        if not hasattr(self, 'cluster_analysis') or not self.cluster_analysis:
            raise ValueError("No hay análisis disponible. Ejecuta analyze_clusters() primero.")
        
        total_entregas = len(self.processed_data)
        total_clusters = len(self.cluster_analysis)
        
        categorias_count = defaultdict(int)
        entregas_por_categoria = defaultdict(int)
        
        for cluster_data in self.cluster_analysis.values():
            cat = cluster_data['categoria']
            tamaño = cluster_data['tamano_grupo']['cantidad']
            categorias_count[cat] += 1
            entregas_por_categoria[cat] += tamaño
        
        score_global = np.mean([
            c['metricas_rendimiento']['score_promedio']['valor'] 
            for c in self.cluster_analysis.values()
        ])
        
        summary = {
            "resumen_general": {
                "total_entregas_analizadas": total_entregas,
                "total_clusters_identificados": total_clusters,
                "score_promedio_global": round(score_global, 2),
                "calificacion_global": self._interpretar_score(score_global)
            },
            "distribucion_por_severidad": {
                "criticos": {
                    "clusters": categorias_count.get('critico', 0),
                    "entregas": entregas_por_categoria.get('critico', 0),
                    "porcentaje": round(entregas_por_categoria.get('critico', 0) / total_entregas * 100, 1)
                },
                "riesgo": {
                    "clusters": categorias_count.get('riesgo', 0),
                    "entregas": entregas_por_categoria.get('riesgo', 0),
                    "porcentaje": round(entregas_por_categoria.get('riesgo', 0) / total_entregas * 100, 1)
                },
                "atencion": {
                    "clusters": categorias_count.get('atencion', 0),
                    "entregas": entregas_por_categoria.get('atencion', 0),
                    "porcentaje": round(entregas_por_categoria.get('atencion', 0) / total_entregas * 100, 1)
                },
                "satisfactorio": {
                    "clusters": categorias_count.get('satisfactorio', 0),
                    "entregas": entregas_por_categoria.get('satisfactorio', 0),
                    "porcentaje": round(entregas_por_categoria.get('satisfactorio', 0) / total_entregas * 100, 1)
                },
                "excelente": {
                    "clusters": categorias_count.get('excelente', 0),
                    "entregas": entregas_por_categoria.get('excelente', 0),
                    "porcentaje": round(entregas_por_categoria.get('excelente', 0) / total_entregas * 100, 1)
                }
            },
            "alertas": []
        }
        
        if entregas_por_categoria.get('critico', 0) > 0:
            summary['alertas'].append({
                "nivel": "CRÍTICO",
                "mensaje": f" {entregas_por_categoria['critico']} entregas requieren atención INMEDIATA"
            })
        
        if entregas_por_categoria.get('riesgo', 0) + entregas_por_categoria.get('critico', 0) > total_entregas * 0.3:
            summary['alertas'].append({
                "nivel": "ALTO",
                "mensaje": f" Más del 30% de las entregas tienen problemas significativos"
            })
        
        return summary
    
    def run_complete_analysis(self, export_json=True):
        """
        Ejecuta el análisis completo de principio a fin - MEJORADO
        """
        print(" Iniciando análisis completo de clustering...")
        
        try:
            self.load_data()
            self.extract_features()
            self.create_feature_matrix()
            
            self.perform_clustering()
            cluster_analysis = self.analyze_clusters()
            
            comparison = self.compare_clusters()
            
            executive_summary = self.get_executive_summary()
            print("\n RESUMEN EJECUTIVO:")
            print(f"   Score Global: {executive_summary['resumen_general']['score_promedio_global']:.2f}/5.0")
            print(f"   Calificación: {executive_summary['resumen_general']['calificacion_global']}")
            
            try:
                self.visualize_clusters()
            except Exception as e:
                print(f" Error generando visualizaciones: {e}")
            
            self.save_model('survey_clustering_model.pkl')
            
            if export_json:
                self.export_analysis_json('cluster_analysis_clean.json')
            
            print("\n ¡Análisis completado exitosamente!")
            
            return {
                'cluster_analysis': cluster_analysis,
                'comparison': comparison,
                'executive_summary': executive_summary,
                'model_saved': True
            }
            
        except Exception as e:
            print(f"\n Error durante el análisis: {str(e)}")
            raise

if __name__ == "__main__":
    analyzer = SurveyAnalyzer('response_1762669055978.json')
    results = analyzer.run_complete_analysis()