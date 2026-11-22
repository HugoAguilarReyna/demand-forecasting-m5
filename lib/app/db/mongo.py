"""Colecciones M5 y conexión local; no depende del backend HTTP."""
from pymongo import MongoClient
from app.comercial.conexion import resolver_destino

M5_IMPORTACIONES = "m5_importaciones"
M5_ESTADO = "m5_estado"
M5_CATALOGO = "m5_catalogo"
M5_CALENDARIO = "m5_calendario"
M5_AGREGADOS = "m5_agregados"
M5_BLOQUES = "m5_bloques"
M5_PRONOSTICOS = "m5_pronosticos"
M5_EVALUACION = "m5_evaluacion"
M5_DERIVADOS = "m5_derivados"
M5_PRECIOS = "m5_precios"
M5_PRECIOS_CATALOGO = "m5_precios_catalogo"
M5_PRECIOS_MANIFEST = "m5_precios_manifest"
M5_MODELOS = "m5_modelos"
M5_PREDICCIONES = "m5_predicciones"
M5_PREDICCIONES_SERIES = "m5_predicciones_series"
M5_EVALUACION_MODELOS = "m5_evaluacion_modelos"
M5_PLANIFICACION = "m5_planificacion"
M5_EVALUACION_SERIES = "m5_evaluacion_series"
M5_AUDITORIA_TEMPORAL = "m5_auditoria_temporal"
M5_PUBLICACION = "m5_publicacion"

class MongoDatabase:
    def __init__(self, settings):
        uri, base = resolver_destino(settings.comercial_mongodb_uri, settings.comercial_mongodb_db)
        self.client = MongoClient(uri, serverSelectionTimeoutMS=settings.mongo_timeout_ms)
        self.m5_db = self.client[base]

    def close(self):
        self.client.close()

def init_database(settings):
    return MongoDatabase(settings)
