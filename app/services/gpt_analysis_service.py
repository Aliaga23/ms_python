from openai import OpenAI
import json
import os
from typing import Dict

class GPTAnalysisService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY no está configurada en las variables de entorno")
            cls._instance.client = OpenAI(api_key=api_key)
            cls._instance.model = "gpt-4o-mini"
        return cls._instance
    
    def analyze_clustering_results(self, clustering_data: Dict) -> Dict:
        prompt = self._build_analysis_prompt(clustering_data)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """Eres un analista experto que traduce resultados técnicos de clustering a lenguaje claro para el público general.
Tu trabajo es:
1. Reemplazar términos técnicos (cluster, silhouette, centroide) por lenguaje comprensible (grupo, segmento, perfil)
2. Crear nombres descriptivos y profesionales para cada grupo (2-5 palabras)
3. Proporcionar interpretaciones claras basadas en estadísticas
4. Dar recomendaciones accionables y específicas
5. Generar el resultado final completo y estructurado para el usuario final

IMPORTANTE:
- NO uses la palabra "cluster", usa "grupo" o "segmento"
- NO uses jerga técnica de machine learning
- Usa un tono profesional pero accesible
- Las métricas deben explicarse en términos de calidad/rendimiento
- Sin emojis

Responde SIEMPRE en formato JSON válido, sin markdown."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return self._build_final_response(clustering_data, result)
            
        except Exception as e:
            return {
                "error": f"Error al analizar con GPT: {str(e)}",
                "datos_originales": clustering_data
            }
    
    def _build_final_response(self, clustering_data: Dict, gpt_analysis: Dict) -> Dict:
        clusters_summary = clustering_data.get("clusters_summary", [])
        
        grupos_mejorados = {}
        comparativa_grupos = []
        
        for cluster_summary in clusters_summary:
            cluster_id = cluster_summary["cluster_id"]
            cluster_id_str = str(cluster_id)
            
            grupo_gpt = gpt_analysis.get("grupos", {}).get(cluster_id_str, {})
            
            grupo_mejorado = {
                "nombre": grupo_gpt.get("nombre", cluster_summary["nombre"]),
                "descripcion": grupo_gpt.get("descripcion", cluster_summary["descripcion"]),
                "cantidad_participantes": cluster_summary["cantidad_personas"],
                "porcentaje": cluster_summary["porcentaje"],
                "especialidad": cluster_summary["especialidad_principal"],
                "metricas": {
                    "promedio_general": cluster_summary["valores_para_graficas"]["promedio_general"],
                    "promedio_minimo": cluster_summary["valores_para_graficas"]["rango_min"],
                    "promedio_maximo": cluster_summary["valores_para_graficas"]["rango_max"],
                    "consistencia": "Alta" if cluster_summary["valores_para_graficas"]["promedio_general"] > 2.0 else "Media" if cluster_summary["valores_para_graficas"]["promedio_general"] > 1.0 else "Baja"
                },
                "respuestas_destacadas": cluster_summary["respuestas_destacadas"],
                "interpretacion": grupo_gpt.get("interpretacion", ""),
                "hallazgos_clave": grupo_gpt.get("hallazgos", []),
                "recomendaciones": grupo_gpt.get("recomendaciones", []),
                "nota": grupo_gpt.get("nota_contexto", "")
            }
            
            grupos_mejorados[f"grupo_{cluster_id + 1}"] = grupo_mejorado
            
            comparativa_grupos.append({
                "nombre": grupo_mejorado["nombre"],
                "cantidad": grupo_mejorado["cantidad_participantes"],
                "porcentaje": grupo_mejorado["porcentaje"],
                "promedio": grupo_mejorado["metricas"]["promedio_general"],
                "especialidad": grupo_mejorado["especialidad"]
            })
        
        comparativa_grupos.sort(key=lambda x: x["promedio"], reverse=True)
        
        calidad_analisis = "Excelente" if clustering_data.get("silhouette_score", 0) > 0.7 else "Buena" if clustering_data.get("silhouette_score", 0) > 0.5 else "Aceptable"
        
        return {
            "encuesta": {
                "nombre": "Análisis de Segmentación Multi-Área",
                "descripcion": "Encuesta sobre rendimiento, seguridad y mantenimiento",
                "tema_principal": "general"
            },
            "resumen_ejecutivo": {
                "total_participantes": clustering_data.get("total_destinatarios", 0),
                "total_preguntas": len([r for cs in clustering_data.get("clusters_summary", []) for r in cs.get("respuestas_destacadas", [])]),
                "preguntas_con_respuestas": len([cs for cs in clustering_data.get("clusters_summary", []) if cs.get("respuestas_destacadas")]),
                "grupos_identificados": clustering_data.get("n_clusters", 0),
                "calidad_segmentacion": calidad_analisis,
                "descripcion_general": gpt_analysis.get("resumen_general", "")
            },
            "estadisticas_generales": {},
            "preguntas_evaluadas": [
                r for cs in clustering_data.get("clusters_summary", []) 
                for r in cs.get("respuestas_destacadas", [])
            ],
            "grupos": grupos_mejorados,
            "comparativa_grupos": comparativa_grupos,
            "conclusiones": {
                "patrones_identificados": gpt_analysis.get("patrones_principales", []),
                "areas_fortaleza": gpt_analysis.get("areas_fortaleza", []),
                "areas_mejora": gpt_analysis.get("areas_mejora", []),
                "recomendaciones_generales": gpt_analysis.get("recomendaciones_generales", [])
            },
            "metadatos": {
                "metodo_usado": "Segmentación automática con análisis inteligente",
                "confiabilidad": calidad_analisis,
                "nota": "Los valores en 0 indican preguntas no respondidas por ese grupo de participantes"
            }
        }
    
    def _build_analysis_prompt(self, data: Dict) -> str:
        n_clusters = data.get("n_clusters", 0)
        silhouette = data.get("silhouette_score", 0)
        clusters_summary = data.get("clusters_summary", [])
        
        calidad_segmentacion = "excelente" if silhouette > 0.7 else "buena" if silhouette > 0.5 else "aceptable"
        
        prompt = f"""Analiza estos resultados de segmentación automática de una encuesta y presenta los resultados para un público NO TÉCNICO.

CONTEXTO GENERAL:
- Total de participantes: {data.get('total_destinatarios', 0)}
- Grupos identificados: {n_clusters}
- Calidad de la segmentación: {calidad_segmentacion}

RESUMEN DE CADA GRUPO (NO uses la palabra "cluster"):
"""
        
        for cluster_summary in clusters_summary:
            grupo_num = cluster_summary.get('cluster_id') + 1
            
            prompt += f"""
GRUPO {grupo_num} - {cluster_summary['nombre']}:
- Descripción: {cluster_summary['descripcion']}
- Participantes: {cluster_summary['cantidad_personas']} ({cluster_summary['porcentaje']}% del total)
- Especialidad: {cluster_summary['especialidad_principal']}
- Promedio general: {cluster_summary['valores_para_graficas']['promedio_general']} (escala aproximada de 1 a 5)
- Rango: {cluster_summary['valores_para_graficas']['rango_min']} - {cluster_summary['valores_para_graficas']['rango_max']}
- Respuestas destacadas: {json.dumps(cluster_summary['respuestas_destacadas'], indent=2)}
"""
        
        prompt += """

RESPONDE EN FORMATO JSON CON ESTA ESTRUCTURA EXACTA (usa lenguaje NO TÉCNICO):
{
  "resumen_general": "Descripción clara del análisis (2-3 oraciones). Menciona que cada grupo respondió diferentes preguntas si aplica.",
  "grupos": {
    "0": {
      "nombre": "Nombre descriptivo basado en lo que SÍ respondieron (ej: 'Enfoque en Seguridad', 'Prioridad en Calidad')",
      "descripcion": "Qué caracteriza a este grupo según sus respuestas (1 oración clara)",
      "nivel_rendimiento": "Excelente / Bueno / Aceptable / Bajo",
      "interpretacion": "Explicación de sus respuestas EN LAS PREGUNTAS QUE RESPONDIERON (2-3 oraciones)",
      "hallazgos": [
        "Hallazgo específico basado en datos reales (no menciones preguntas con valor 0)",
        "Hallazgo específico 2"
      ],
      "recomendaciones": [
        "Recomendación accionable basada en lo que respondieron",
        "Recomendación específica 2"
      ],
      "nota_contexto": "Breve nota sobre qué preguntas/áreas evaluó este grupo"
    },
    "1": { ... },
    ...
  },
  "patrones_principales": [
    "Patrón importante 1 (enfócate en diferencias entre grupos)",
    "Patrón importante 2"
  ],
  "areas_fortaleza": [
    "Área fuerte identificada en los datos",
    "Área fuerte 2"
  ],
  "areas_mejora": [
    "Área que necesita atención según las respuestas",
    "Área de mejora 2"
  ],
  "recomendaciones_generales": [
    "Recomendación global 1 basada en todos los grupos",
    "Recomendación global 2",
    "Recomendación global 3"
  ]
}

REGLAS CRÍTICAS:
- IGNORA completamente las preguntas con promedio 0 - significa que NO las respondieron
- Enfoca tu análisis SOLO en lo que cada grupo SÍ respondió (valores > 0)
- Si un grupo respondió pocas preguntas, menciónalo en "nota_contexto"
- Los nombres de grupos deben reflejar QUÉ evaluaron (ej: "Grupo de Seguridad", "Grupo de Rendimiento")
- NO uses "cluster", "clustering", "silhouette", "centroide"
- Lenguaje claro y profesional para público general
- Sin emojis, sin markdown, solo JSON válido"""
        
        return prompt
