"""Modelo global de árboles para el pronóstico diario a 28 días (Walmart M5).

Qué hace este módulo
--------------------
1. Carga las ventas **artículo × tienda** (``m5_bloques``) en una matriz, el
   calendario y los precios importados.
2. Construye variables **con información disponible en el origen**: rezagos,
   estadísticas móviles causales, calendario/eventos/SNAP del día objetivo y precios
   conocidos hasta la semana del origen.
3. Entrena un modelo global de árboles (LightGBM) **condicionado al horizonte**
   (``h = 1…28`` es una variable más), de modo que los días 8 a 28 se generan con
   variables del origen y del calendario, nunca con ventas reales futuras.
4. Pronostica, compara contra el baseline estacional semanal y evalúa por horizonte
   y por segmento, distinguiendo el error a grano artículo × tienda del error de las
   series agregadas.

Política de disponibilidad de variables
---------------------------------------
===============================  ==========================================
Variable                          Disponibilidad
===============================  ==========================================
Ventas pasadas (rezagos, medias)  Conocida al origen (solo ``t ≤ O``)
Calendario del día objetivo       Conocida anticipadamente (mes, día de la
                                  semana, semana del año, eventos, SNAP)
SNAP de la tienda                 Flag real del calendario para ese estado
Precio hasta la semana del origen Conocida al origen (último precio conocido)
Precio del día objetivo           **Supuesta**: se mantiene el último precio
                                  conocido al origen (no hay precios futuros)
===============================  ==========================================

Nunca se rellenan precios históricos con observaciones posteriores, y el precio
futuro no se toma del archivo de precios como si fuera conocido.

Transformaciones
----------------
- Las predicciones negativas se recortan a cero **en la etapa de predicción**, y la
  misma transformación se aplica al evaluar.
- No se redondea antes de calcular métricas; el redondeo se hace solo al publicar el
  vector de unidades enteras.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from pymongo.database import Database

from app.comercial import fuentes, precios as precios_mod
from app.db.mongo import (
    M5_BLOQUES,
    M5_CALENDARIO,
    M5_CATALOGO,
    M5_MODELOS,
    M5_PREDICCIONES,
    M5_PREDICCIONES_SERIES,
)

HORIZONTE = fuentes.DIAS_HORIZONTE  # 28
SEMILLA = 20260522

VARIANTES = {
    "calendario": {
        "etiqueta": "Rezagos + calendario, eventos y SNAP",
        "precios": False,
        "descripcion": "Modelo global de árboles con historia de ventas y calendario; sin precios.",
    },
    "precios": {
        "etiqueta": "Rezagos + calendario + precios",
        "precios": True,
        "descripcion": "Añade el precio conocido al origen y sus cambios recientes.",
    },
}

ARTEFACTOS = Path(os.environ.get("COMERCIAL_ARTIFACT_DIR", str(Path.cwd() / "data" / "artefactos")))

#: Cortes reservados: los dos últimos no participan en la selección del modelo.
ORIGENES_DESARROLLO = (1773, 1801, 1829, 1857)
ORIGENES_FINALES = (1885, 1913)
MINIMO_HISTORIA = 365 + 28
"""Se exige más de un año de historia para poder usar el rezago anual."""


@dataclass
class Contexto:
    """Datos del modelo en memoria (ventas, calendario y precios alineados)."""

    version: str
    version_precios: str | None
    series: list[tuple[str, str]]
    item_idx: np.ndarray
    store_idx: np.ndarray
    cat_idx: np.ndarray
    dept_idx: np.ndarray
    state_idx: np.ndarray
    ventas: np.ndarray
    calendario: dict[str, np.ndarray]
    semana_por_dia: np.ndarray
    semanas: list[int]
    precios_serie: np.ndarray | None
    precios_rellenados: np.ndarray | None
    pos_semana: dict[int, int] = field(default_factory=dict)
    fecha_de_dia: list[date] = field(default_factory=list)
    nombres: dict[str, list[str]] = field(default_factory=dict)
    metricas_carga: dict = field(default_factory=dict)

    @property
    def n_series(self) -> int:
        return len(self.series)

    @property
    def n_dias(self) -> int:
        return int(self.ventas.shape[1])


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #


def _ruta_cache(version: str) -> Path:
    return ARTEFACTOS / f"ventas_{version}.npy"


def _cargar_ventas(db: Database, version: str) -> np.ndarray:
    """Matriz ``(series, días)`` int32 desde los bloques artículo × tienda."""
    catalogo = list(
        db[M5_CATALOGO].find(
            {"version": version, "tipo": "producto"}, {"item_id": 1, "dept_id": 1, "cat_id": 1}
        )
    )
    tiendas = sorted(
        d["store_id"] for d in db[M5_CATALOGO].find({"version": version, "tipo": "tienda"}, {"store_id": 1})
    )
    total = db.m5_agregados.find_one({"version": version, "dimension": "total"}, {"unidades": 1})
    if not total:
        raise RuntimeError("No hay agregados de la versión publicada")
    n_dias = len(total["unidades"])
    items = sorted(d["item_id"] for d in catalogo)
    pos_item = {item: indice for indice, item in enumerate(items)}
    pos_tienda = {tienda: indice for indice, tienda in enumerate(tiendas)}
    matriz = np.zeros((len(items) * len(tiendas), n_dias), dtype=np.int32)
    indice = {
        (item, tienda): pos_item[item] * len(tiendas) + pos_tienda[tienda]
        for item in items
        for tienda in tiendas
    }
    for documento in db[M5_BLOQUES].find(
        {"version": version}, {"item_id": 1, "store_id": 1, "d_inicio": 1, "unidades": 1}
    ):
        fila = indice[(documento["item_id"], documento["store_id"])]
        inicio = int(documento["d_inicio"]) - 1
        unidades = documento["unidades"]
        matriz[fila, inicio : inicio + len(unidades)] = unidades
    return matriz


def cargar_contexto(
    db: Database,
    version: str,
    *,
    version_precios: str | None = None,
    usar_cache: bool = True,
    log=None,
) -> Contexto:
    """Prepara ventas, calendario y precios para el modelo (con caché en disco)."""
    inicio = time.perf_counter()
    ARTEFACTOS.mkdir(parents=True, exist_ok=True)
    ruta = _ruta_cache(version)
    if usar_cache and ruta.exists():
        ventas = np.load(ruta)
        origen_ventas = "caché en disco"
    else:
        ventas = _cargar_ventas(db, version)
        np.save(ruta, ventas)
        origen_ventas = "m5_bloques"

    catalogo = list(
        db[M5_CATALOGO].find(
            {"version": version, "tipo": "producto"}, {"item_id": 1, "dept_id": 1, "cat_id": 1}
        )
    )
    tiendas = list(
        db[M5_CATALOGO].find({"version": version, "tipo": "tienda"}, {"store_id": 1, "state_id": 1})
    )
    tiendas.sort(key=lambda d: d["store_id"])
    items = sorted(d["item_id"] for d in catalogo)
    estados = sorted({d["state_id"] for d in tiendas})
    categorias = sorted({d["cat_id"] for d in catalogo})
    departamentos = sorted({d["dept_id"] for d in catalogo})
    pos_item = {item: indice for indice, item in enumerate(items)}
    pos_tienda = {tienda["store_id"]: indice for indice, tienda in enumerate(tiendas)}
    pos_cat = {categoria: indice for indice, categoria in enumerate(categorias)}
    pos_dept = {departamento: indice for indice, departamento in enumerate(departamentos)}
    pos_estado = {estado: indice for indice, estado in enumerate(estados)}

    series: list[tuple[str, str]] = []
    item_idx, store_idx, cat_idx, dept_idx, state_idx = [], [], [], [], []
    for item in items:
        meta = next(d for d in catalogo if d["item_id"] == item)
        for tienda in tiendas:
            series.append((item, tienda["store_id"]))
            item_idx.append(pos_item[item])
            store_idx.append(pos_tienda[tienda["store_id"]])
            cat_idx.append(pos_cat[meta["cat_id"]])
            dept_idx.append(pos_dept[meta["dept_id"]])
            state_idx.append(pos_estado[tienda["state_id"]])

    # El orden de la matriz de ventas es (item, tienda) con tienda como eje rápido.
    orden = np.array(
        [
            pos_item[item] * len(tiendas) + pos_tienda[tienda]
            for item, tienda in series
        ],
        dtype=np.int64,
    )
    ventas = ventas[orden]

    dias = list(db[M5_CALENDARIO].find({}, {"_id": 0}).sort("d", 1))
    calendario = {
        "d": np.array([int(d["d"]) for d in dias], dtype=np.int32),
        "dow": np.array([int(d["wday"]) for d in dias], dtype=np.int8),
        "month": np.array([int(d["month"]) for d in dias], dtype=np.int8),
        "year": np.array([int(d["year"]) for d in dias], dtype=np.int16),
        "day": np.array([int(d["fecha"].day) for d in dias], dtype=np.int8),
        "week": np.array([int(d["wm_yr_wk"]) for d in dias], dtype=np.int32),
        "snap_ca": np.array([int(d.get("snap_ca") or 0) for d in dias], dtype=np.int8),
        "snap_tx": np.array([int(d.get("snap_tx") or 0) for d in dias], dtype=np.int8),
        "snap_wi": np.array([int(d.get("snap_wi") or 0) for d in dias], dtype=np.int8),
        "evento_1": np.array([0 if not d.get("evento_1") else 1 for d in dias], dtype=np.int8),
        "evento_2": np.array([0 if not d.get("evento_2") else 1 for d in dias], dtype=np.int8),
    }
    tipos = sorted({str(d.get("tipo_1") or "") for d in dias} | {str(d.get("tipo_2") or "") for d in dias})
    pos_tipo = {tipo: indice for indice, tipo in enumerate(tipos)}
    calendario["tipo_1"] = np.array([pos_tipo[str(d.get("tipo_1") or "")] for d in dias], dtype=np.int8)
    calendario["tipo_2"] = np.array([pos_tipo[str(d.get("tipo_2") or "")] for d in dias], dtype=np.int8)
    fechas = [d["fecha"].date() if isinstance(d["fecha"], datetime) else d["fecha"] for d in dias]
    anio_inicio = min(fechas)
    calendario["doy"] = np.array([(fecha - date(fecha.year, 1, 1)).days + 1 for fecha in fechas], dtype=np.int16)
    calendario["semana_anio"] = np.array([int(fecha.isocalendar()[1]) for fecha in fechas], dtype=np.int8)
    calendario["fin_de_semana"] = np.isin(calendario["dow"], (1, 7)).astype(np.int8)
    calendario["desde_inicio"] = np.array(
        [(fecha - anio_inicio).days for fecha in fechas], dtype=np.int32
    )
    del anio_inicio

    semanas = sorted({int(d["wm_yr_wk"]) for d in dias})
    pos_semana = {semana: indice for indice, semana in enumerate(semanas)}
    semana_por_dia = np.array([pos_semana[int(d["wm_yr_wk"])] for d in dias], dtype=np.int32)

    precios_serie = None
    precios_rellenados = None
    if version_precios:
        cargados = precios_mod.cargar_precios(db, version_precios)
        pos_item_precio = {item: indice for indice, item in enumerate(cargados["items"])}
        pos_tienda_precio = {tienda: indice for indice, tienda in enumerate(cargados["tiendas"])}
        semanas_precio = cargados["semanas"]
        matriz = cargados["matriz"]
        # Se indexa por semana del calendario: las semanas del archivo de precios son
        # las mismas del calendario (se validó al importar).
        pos_semana_precio = {semana: indice for indice, semana in enumerate(semanas_precio)}
        faltantes = [semana for semana in semanas if semana not in pos_semana_precio]
        if faltantes:
            raise RuntimeError(f"Semanas del calendario sin precio importado: {faltantes[:3]}")
        i_item = np.array([pos_item_precio[item] for item, _ in series], dtype=np.int64)
        i_tienda = np.array([pos_tienda_precio[tienda] for _, tienda in series], dtype=np.int64)
        i_semana = np.array([pos_semana_precio[semana] for semana in semanas], dtype=np.int64)
        # Índice por serie (no producto cartesiano): (n_semanas, n_series) → traspuesto.
        precios_serie = matriz[
            i_semana[:, None], i_item[None, :], i_tienda[None, :]
        ].T.astype(np.float32)
        # Relleno causal: cada semana hereda el último precio conocido **anterior**.
        precios_rellenados = _rellenar_causal(precios_serie)

    metricas_carga = {
        "origen_ventas": origen_ventas,
        "n_series": len(series),
        "n_dias": int(ventas.shape[1]),
        "n_semanas": len(semanas),
        "memoria_ventas_mb": round(ventas.nbytes / 1024**2, 1),
        "memoria_precios_mb": round(precios_serie.nbytes / 1024**2, 1) if precios_serie is not None else 0.0,
        "duracion_s": round(time.perf_counter() - inicio, 1),
    }
    if log:
        log(metricas_carga)

    return Contexto(
        version=version,
        version_precios=version_precios,
        series=series,
        item_idx=np.array(item_idx, dtype=np.int32),
        store_idx=np.array(store_idx, dtype=np.int32),
        cat_idx=np.array(cat_idx, dtype=np.int8),
        dept_idx=np.array(dept_idx, dtype=np.int8),
        state_idx=np.array(state_idx, dtype=np.int8),
        ventas=ventas,
        calendario=calendario,
        semana_por_dia=semana_por_dia,
        semanas=semanas,
        precios_serie=precios_serie,
        precios_rellenados=precios_rellenados,
        pos_semana=pos_semana,
        fecha_de_dia=fechas,
        nombres={
            "items": items,
            "tiendas": [d["store_id"] for d in tiendas],
            "estados": estados,
            "categorias": categorias,
            "departamentos": departamentos,
            "tipos_evento": tipos,
        },
        metricas_carga=metricas_carga,
    )


def _rellenar_causal(precios: np.ndarray) -> np.ndarray:
    """Rellena hacia adelante con el último precio conocido (nunca hacia atrás)."""
    salida = np.array(precios, dtype=np.float32, copy=True)
    for semana in range(1, salida.shape[1]):
        faltante = np.isnan(salida[:, semana])
        salida[faltante, semana] = salida[faltante, semana - 1]
    return salida


# --------------------------------------------------------------------------- #
# Variables
# --------------------------------------------------------------------------- #

FEATURES_BASE = (
    "h",
    "dow",
    "month",
    "day",
    "doy",
    "semana_anio",
    "fin_de_semana",
    "evento_1",
    "evento_2",
    "tipo_1",
    "tipo_2",
    "snap",
    "desde_inicio",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "lag_5",
    "lag_6",
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    "lag_364",
    "media_7",
    "media_14",
    "media_28",
    "media_56",
    "media_112",
    "std_7",
    "std_28",
    "std_56",
    "dow_media_1",
    "dow_media_2",
    "dow_media_3",
    "dow_media_4",
    "dow_media_8",
    "ceros_28",
    "ceros_56",
    "ultimo_positivo",
    "dias_desde_venta",
    "positivos_28",
    "ratio_7_28",
    "ratio_lag7_media28",
    "ratio_28_112",
    "tienda",
    "estado",
    "categoria",
    "departamento",
    "item",
)

FEATURES_PRECIO = (
    "precio_origen",
    "precio_medio_4s",
    "precio_medio_26s",
    "precio_rel_26",
    "precio_cambio_4s",
    "semanas_sin_cambio",
    "tiene_precio",
    "precio_supuesto",
    "precio_supuesto_rel",
)

CATEGORICAS = ("tienda", "estado", "categoria", "departamento", "item", "tipo_1", "tipo_2", "dow")


def _features(con_precios: bool) -> tuple[str, ...]:
    return FEATURES_BASE + (FEATURES_PRECIO if con_precios else ())


def construir_matriz(
    ctx: Contexto, origen: int, *, con_precios: bool
) -> tuple[np.ndarray, list[str]]:
    """Matriz de variables para ``origen`` (día observado, 1-based) y h = 1..28.

    Todas las variables de ventas usan ``t ≤ origen``; las del calendario y SNAP
    corresponden al día objetivo (información conocida anticipadamente); el precio
    del día objetivo es una **suposición** explícita (último precio conocido).
    """
    # La historia debe existir (``origen`` es un día observado), pero los días
    # objetivo pueden caer en el calendario posterior a las ventas: es justo el caso
    # del pronóstico final, y el calendario sí los cubre.
    if origen < MINIMO_HISTORIA or origen > ctx.n_dias:
        raise ValueError(f"Origen fuera de rango: {origen}")
    if origen + HORIZONTE > int(ctx.calendario["d"].size):
        raise ValueError("El calendario no cubre los 28 días objetivo del origen")
    nombres = list(_features(con_precios))
    n = ctx.n_series
    H = HORIZONTE
    columnas: dict[str, np.ndarray] = {}

    historia = ctx.ventas[:, :origen].astype(np.float32)
    cal = ctx.calendario
    # Índices del día objetivo: t = origen - 1 + h (0-based) para h = 1..28.
    objetivo = (origen - 1) + np.arange(1, H + 1)

    def por_horizonte(valores: np.ndarray) -> np.ndarray:
        """Repite un vector por serie a lo largo de los 28 horizontes."""
        return np.repeat(valores.astype(np.float32)[:, None], H, axis=1)

    columnas["h"] = np.tile(np.arange(1, H + 1, dtype=np.float32), (n, 1))
    for nombre, arreglo in (
        ("dow", cal["dow"]),
        ("month", cal["month"]),
        ("day", cal["day"]),
        ("doy", cal["doy"]),
        ("semana_anio", cal["semana_anio"]),
        ("fin_de_semana", cal["fin_de_semana"]),
        ("evento_1", cal["evento_1"]),
        ("evento_2", cal["evento_2"]),
        ("tipo_1", cal["tipo_1"]),
        ("tipo_2", cal["tipo_2"]),
        ("desde_inicio", cal["desde_inicio"]),
    ):
        columnas[nombre] = np.tile(arreglo[objetivo].astype(np.float32), (n, 1))

    snap_estado = np.stack([cal["snap_ca"], cal["snap_tx"], cal["snap_wi"]], axis=1)
    # SNAP real del estado de cada tienda en el día objetivo (flag del calendario).
    columnas["snap"] = snap_estado[np.ix_(objetivo, ctx.state_idx)].T.astype(np.float32)

    # Rezagos (t ≤ origen). Si el rezago sale antes del inicio se deja 0 y se marca
    # con los indicadores de "sin historia".
    for rezago in (1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 364):
        if rezago <= origen:
            columnas[f"lag_{rezago}"] = por_horizonte(historia[:, -rezago])
        else:
            columnas[f"lag_{rezago}"] = por_horizonte(np.zeros(n, dtype=np.float32))

    ventana = lambda k: historia[:, -k:]  # noqa: E731
    for k in (7, 14, 28, 56, 112):
        if k <= origen:
            columnas[f"media_{k}"] = por_horizonte(ventana(k).mean(axis=1))
        else:
            columnas[f"media_{k}"] = por_horizonte(np.zeros(n, dtype=np.float32))
    for k in (7, 28, 56):
        if k <= origen:
            columnas[f"std_{k}"] = por_horizonte(ventana(k).std(axis=1))
        else:
            columnas[f"std_{k}"] = por_horizonte(np.zeros(n, dtype=np.float32))

    # Media del mismo día de la semana: posición O-j*7 (j = 1..8).
    for j in (1, 2, 3, 4):
        indice = origen - 7 * j
        if indice >= 0:
            columnas[f"dow_media_{j}"] = por_horizonte(historia[:, indice])
        else:
            columnas[f"dow_media_{j}"] = por_horizonte(np.zeros(n, dtype=np.float32))
    indices = [origen - 7 * j for j in range(1, 9) if origen - 7 * j >= 0]
    if indices:
        columnas["dow_media_8"] = por_horizonte(historia[:, indices].mean(axis=1))
    else:
        columnas["dow_media_8"] = por_horizonte(np.zeros(n, dtype=np.float32))

    positivos = historia > 0
    for k in (28, 56):
        if k <= origen:
            columnas[f"ceros_{k}"] = por_horizonte(1.0 - positivos[:, -k:].mean(axis=1))
        else:
            columnas[f"ceros_{k}"] = por_horizonte(np.zeros(n, dtype=np.float32))
    columnas["positivos_28"] = por_horizonte(positivos[:, -28:].sum(axis=1).astype(np.float32))

    # Último valor positivo y días desde la última venta (intermitencia).
    invertido = positivos[:, ::-1]
    encontrado = invertido.any(axis=1)
    posiciones = np.where(encontrado, origen - 1 - np.argmax(invertido, axis=1), -1)
    ultimo = np.zeros(n, dtype=np.float32)
    dias_desde = np.full(n, 365.0, dtype=np.float32)
    filas = np.where(encontrado)[0]
    if filas.size:
        ultimo[filas] = historia[filas, posiciones[filas]]
        dias_desde[filas] = origen - 1 - posiciones[filas]
    columnas["ultimo_positivo"] = por_horizonte(ultimo)
    columnas["dias_desde_venta"] = por_horizonte(dias_desde)

    media_7 = columnas["media_7"][:, 0]
    media_28 = columnas["media_28"][:, 0]
    media_112 = columnas["media_112"][:, 0]
    lag_7 = columnas["lag_7"][:, 0]
    seguro = lambda v: np.where(np.abs(v) < 1e-6, np.nan, v)  # noqa: E731
    columnas["ratio_7_28"] = por_horizonte(np.nan_to_num(media_7 / seguro(media_28), nan=1.0))
    columnas["ratio_lag7_media28"] = por_horizonte(np.nan_to_num(lag_7 / seguro(media_28), nan=1.0))
    columnas["ratio_28_112"] = por_horizonte(np.nan_to_num(media_28 / seguro(media_112), nan=1.0))

    columnas["tienda"] = por_horizonte(ctx.store_idx)
    columnas["estado"] = por_horizonte(ctx.state_idx)
    columnas["categoria"] = por_horizonte(ctx.cat_idx)
    columnas["departamento"] = por_horizonte(ctx.dept_idx)
    columnas["item"] = por_horizonte(ctx.item_idx)

    if con_precios:
        if ctx.precios_rellenados is None:
            raise RuntimeError("La variante con precios necesita precios importados")
        semana = int(ctx.semana_por_dia[origen - 1])
        precios_hist = ctx.precios_rellenados[:, : semana + 1]
        precio_origen = precios_hist[:, -1]
        tiene = (~np.isnan(ctx.precios_serie[:, semana])).astype(np.float32)
        precio_origen = np.where(np.isnan(precio_origen), np.nan, precio_origen)
        columnas["tiene_precio"] = por_horizonte(tiene)
        # Sin precio conocido se usa la mediana histórica del propio artículo-tienda
        # (nunca un precio futuro) y, si tampoco hay, cero explícito.
        mediana = np.nanmedian(precios_hist, axis=1) if precios_hist.size else np.zeros(n, dtype=np.float32)
        mediana = np.where(np.isnan(mediana), 0.0, mediana)
        precio_origen_lleno = np.where(np.isnan(precio_origen), mediana, precio_origen)
        columnas["precio_origen"] = por_horizonte(precio_origen_lleno)
        columnas["precio_medio_4s"] = por_horizonte(
            np.nanmean(precios_hist[:, -4:], axis=1)
        )
        columnas["precio_medio_26s"] = por_horizonte(
            np.nanmean(precios_hist[:, -26:], axis=1)
        )
        media_26 = columnas["precio_medio_26s"][:, 0]
        columnas["precio_rel_26"] = por_horizonte(
            np.nan_to_num(precio_origen_lleno / seguro(media_26), nan=1.0)
        )
        if precios_hist.shape[1] >= 5:
            hace_4 = precios_hist[:, -5]
            columnas["precio_cambio_4s"] = por_horizonte(
                np.nan_to_num(precio_origen_lleno / seguro(hace_4) - 1.0, nan=0.0)
            )
        else:
            columnas["precio_cambio_4s"] = por_horizonte(np.zeros(n, dtype=np.float32))
        # Semanas consecutivas sin cambio de precio, contando hacia atrás desde el origen.
        cambios = np.diff(precios_hist, axis=1) != 0
        sin_cambio = np.zeros(n, dtype=np.float32)
        activo = np.ones(n, dtype=bool)
        for columna in range(cambios.shape[1] - 1, -1, -1):
            activo &= ~cambios[:, columna]
            sin_cambio[activo] += 1
        columnas["semanas_sin_cambio"] = por_horizonte(sin_cambio)
        # Supuesto: el precio del día objetivo es el último precio conocido.
        columnas["precio_supuesto"] = por_horizonte(precio_origen_lleno)
        columnas["precio_supuesto_rel"] = por_horizonte(
            np.nan_to_num(precio_origen_lleno / seguro(media_26), nan=1.0)
        )

    matriz = np.stack([np.asarray(columnas[nombre], dtype=np.float32) for nombre in nombres], axis=-1)
    return matriz.reshape(n * H, len(nombres)), nombres


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #


def predecir_baseline(ctx: Contexto, origen: int) -> np.ndarray:
    """Estacional semanal: repite los últimos siete días observados."""
    patron = ctx.ventas[:, origen - 7 : origen].astype(np.float32)
    if patron.shape[1] != 7:
        raise ValueError("El origen no tiene siete días observados")
    indices = np.arange(HORIZONTE) % 7
    return patron[:, indices]


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #


def entrenar(
    ctx: Contexto,
    origenes: list[int],
    *,
    con_precios: bool,
    fraccion_ceros: float = 0.06,
    semilla: int = SEMILLA,
    num_boost_round: int = 320,
    objetivo: str = "regression_l2",
    log=None,
) -> dict:
    """Entrena un modelo global condicionado al horizonte.

    El muestreo solo afecta al **entrenamiento**: se conservan todas las filas con
    ventas positivas y una fracción de las ceros, que se reponderan para mantener la
    calibración. La evaluación usa todas las filas.
    """
    import lightgbm as lgb

    inicio = time.perf_counter()
    generador = np.random.default_rng(semilla)
    partes_x, partes_y, partes_w = [], [], []
    for origen in origenes:
        x, nombres = construir_matriz(ctx, origen, con_precios=con_precios)
        y = ctx.ventas[:, origen : origen + HORIZONTE].reshape(-1).astype(np.float32)
        positivos = y > 0
        ceros = np.flatnonzero(~positivos)
        elegidos = generador.random(ceros.size) < fraccion_ceros
        indices = np.concatenate([np.flatnonzero(positivos), ceros[elegidos]])
        pesos = np.ones(indices.size, dtype=np.float32)
        pesos[y[indices] <= 0] = 1.0 / max(fraccion_ceros, 1e-6)
        partes_x.append(x[indices])
        partes_y.append(y[indices])
        partes_w.append(pesos)
        if log:
            log(
                {
                    "origen": origen,
                    "filas_totales": int(x.shape[0]),
                    "filas_muestra": int(indices.size),
                    "positivos": int(positivos.sum()),
                }
            )
        del x
    x_entrenamiento = np.concatenate(partes_x)
    y_entrenamiento = np.concatenate(partes_y)
    w_entrenamiento = np.concatenate(partes_w)
    del partes_x, partes_y, partes_w

    parametros = {
        "objective": objetivo,
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "num_threads": 8,
        "seed": semilla,
        "force_row_wise": True,
        "verbose": -1,
    }
    if objetivo == "tweedie":
        parametros["tweedie_variance_power"] = 1.1
    dataset = lgb.Dataset(
        x_entrenamiento,
        label=y_entrenamiento,
        weight=w_entrenamiento,
        feature_name=nombres,
        categorical_feature=[n for n in nombres if n in CATEGORICAS],
        free_raw_data=True,
    )
    booster = lgb.train(parametros, dataset, num_boost_round=num_boost_round)
    importancias = booster.feature_importance(importance_type="gain")
    orden = np.argsort(importancias)[::-1]
    top = [
        {"variable": nombres[i], "ganancia": round(float(importancias[i]), 1)}
        for i in orden[:12]
        if importancias[i] > 0
    ]
    resultado = {
        "booster": booster,
        "nombres": nombres,
        "variables": _features(con_precios),
        "parametros": parametros,
        "num_boost_round": num_boost_round,
        "filas_entrenamiento": int(x_entrenamiento.shape[0]),
        "origenes_entrenamiento": list(origenes),
        "fraccion_ceros": fraccion_ceros,
        "importancias": top,
        "duracion_s": round(time.perf_counter() - inicio, 1),
        "memoria_mb": round(
            x_entrenamiento.nbytes / 1024**2 + y_entrenamiento.nbytes / 1024**2, 1
        ),
    }
    del x_entrenamiento, y_entrenamiento, w_entrenamiento
    return resultado


def predecir(
    ctx: Contexto, modelo: dict, origen: int, *, baseline: np.ndarray | None = None
) -> tuple[np.ndarray, dict]:
    """Predicción del modelo para las 30 490 series y 28 días (recortada a cero).

    Si una serie produjera algún valor no finito, se aplica **fallback explícito** al
    baseline para esa serie completa: nunca se publica un hueco ni se inventa un valor.
    Devuelve la matriz de predicciones y el informe de cobertura del candidato.
    """
    x, _ = construir_matriz(ctx, origen, con_precios="precio_origen" in modelo["variables"])
    crudo = modelo["booster"].predict(x, num_threads=8)
    del x
    prediccion = np.clip(crudo.reshape(ctx.n_series, HORIZONTE), 0.0, None)
    no_finito = ~np.isfinite(prediccion)
    indices = np.unique(np.flatnonzero(no_finito.any(axis=1)))
    aplicado = 0
    if indices.size:
        if baseline is not None:
            prediccion[indices] = baseline[indices]
            aplicado = int(indices.size)
        else:
            prediccion[no_finito] = 0.0
    informe = {
        "series": int(ctx.n_series),
        "series_modelo": int(ctx.n_series - aplicado),
        "fallback": aplicado,
        "cobertura_candidato_pct": round(100 * (ctx.n_series - aplicado) / ctx.n_series, 2),
        "cobertura_sistema_pct": 100.0,
        "motivo_fallback": (
            None
            if aplicado == 0
            else "Predicción no finita en alguna de las 28 fechas: se usa el baseline de esa serie."
        ),
    }
    return prediccion, informe


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #


def metricas(error: np.ndarray, observado: np.ndarray) -> dict:
    """MAE, sesgo y WAPE a partir de errores y observaciones alineados.

    ``error = predicción − observado``. El WAPE se devuelve como ``None`` cuando el
    denominador es cero (no calculable, nunca cero).
    """
    error = np.asarray(error, dtype=np.float64).reshape(-1)
    observado = np.asarray(observado, dtype=np.float64).reshape(-1)
    validos = ~np.isnan(observado)
    error, observado = error[validos], observado[validos]
    n = int(error.size)
    if n == 0:
        return {
            "n": 0,
            "mae": None,
            "sesgo": None,
            "wape": None,
            "denominador": 0,
            "denominador_cero": True,
        }
    absolutos = float(np.abs(error).sum())
    denominador = float(observado.sum())
    return {
        "n": n,
        "mae": round(absolutos / n, 4),
        "sesgo": round(float(error.sum()) / n, 4),
        "wape": round(100 * absolutos / denominador, 4) if denominador > 0 else None,
        "denominador": int(round(denominador)),
        "denominador_cero": denominador <= 0,
    }


def _mejora(wape_modelo: float | None, wape_base: float | None) -> float | None:
    """Mejora porcentual del error frente al baseline (positivo = mejor)."""
    if wape_modelo is None or wape_base is None or wape_base <= 0:
        return None
    return round(100 * (wape_base - wape_modelo) / wape_base, 3)


def evaluar_origen(
    ctx: Contexto,
    *,
    origen: int,
    prediccion: np.ndarray,
    baseline: np.ndarray,
    etiquetas: dict[str, np.ndarray],
) -> dict:
    """Evalúa candidato y baseline en el mismo origen, serie y fechas.

    Devuelve métricas al grano artículo × tienda, por horizonte, por segmento y de
    las **series agregadas** (sumando predicción y observado antes de medir).
    """
    observado = ctx.ventas[:, origen : origen + HORIZONTE].astype(np.float32)
    valido = np.ones_like(observado, dtype=bool)
    cand = prediccion.copy()
    base = baseline.astype(np.float32)
    errores_cand = cand - observado
    errores_base = base - observado

    resultado: dict = {
        "origen_d": origen,
        "origen_fecha": ctx.fecha_de_dia[origen - 1].isoformat(),
        "fin_fecha": ctx.fecha_de_dia[origen + HORIZONTE - 1].isoformat(),
        "series": int(ctx.n_series),
        "observaciones": int(valido.size),
        "candidato": metricas(np.where(valido, errores_cand, np.nan), observado),
        "baseline": metricas(np.where(valido, errores_base, np.nan), observado),
    }
    resultado["mejora_pct"] = _mejora(
        resultado["candidato"]["wape"], resultado["baseline"]["wape"]
    )
    resultado["mejora_mae_pct"] = _mejora(
        resultado["candidato"]["mae"], resultado["baseline"]["mae"]
    )

    # Por horizonte (día 1…28).
    horizontes = []
    for h in range(HORIZONTE):
        horizontes.append(
            {
                "horizonte": h + 1,
                "fecha": ctx.fecha_de_dia[origen + h].isoformat(),
                "candidato": metricas(errores_cand[:, h], observado[:, h]),
                "baseline": metricas(errores_base[:, h], observado[:, h]),
            }
        )
    resultado["por_horizonte"] = horizontes

    # Por segmento, a grano artículo × tienda y en series agregadas.
    segmentos = []
    for nivel, valores in etiquetas.items():
        for codigo in np.unique(valores):
            mascara = valores == codigo
            cand_seg = metricas(errores_cand[mascara], observado[mascara])
            base_seg = metricas(errores_base[mascara], observado[mascara])
            # Serie agregada del segmento: se suma por día antes de medir.
            cand_agregado = cand[mascara].sum(axis=0)
            base_agregado = base[mascara].sum(axis=0)
            obs_agregado = observado[mascara].sum(axis=0)
            segmentos.append(
                {
                    "nivel": nivel,
                    "clave": str(codigo),
                    "series": int(mascara.sum()),
                    "candidato": cand_seg,
                    "baseline": base_seg,
                    "mejora_pct": _mejora(cand_seg["wape"], base_seg["wape"]),
                    "agregado": {
                        "candidato": metricas(cand_agregado - obs_agregado, obs_agregado),
                        "baseline": metricas(base_agregado - obs_agregado, obs_agregado),
                        "mejora_pct": _mejora(
                            metricas(cand_agregado - obs_agregado, obs_agregado)["wape"],
                            metricas(base_agregado - obs_agregado, obs_agregado)["wape"],
                        ),
                    },
                }
            )
    resultado["segmentos"] = segmentos
    comparables = [s for s in segmentos if s["mejora_pct"] is not None]
    resultado["segmentos_mejoran"] = sum(1 for s in comparables if s["mejora_pct"] > 0)
    resultado["segmentos_empeoran"] = sum(1 for s in comparables if s["mejora_pct"] < 0)
    resultado["segmentos_sin_mejora_calculable"] = len(segmentos) - len(comparables)
    return resultado


def etiquetas_serie(ctx: Contexto) -> dict[str, np.ndarray]:
    """Etiquetas de segmento por serie para la evaluación."""
    return {
        "estado": np.array([ctx.nombres["estados"][i] for i in ctx.state_idx]),
        "tienda": np.array([ctx.nombres["tiendas"][i] for i in ctx.store_idx]),
        "categoria": np.array([ctx.nombres["categorias"][i] for i in ctx.cat_idx]),
        "departamento": np.array([ctx.nombres["departamentos"][i] for i in ctx.dept_idx]),
    }


def agregar_por_dimension(
    ctx: Contexto, valores: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    """Suma un arreglo ``(series, 28)`` por cada dimensión de navegación."""
    tiendas = np.array([ctx.nombres["tiendas"][i] for i in ctx.store_idx])
    estados = np.array([ctx.nombres["estados"][i] for i in ctx.state_idx])
    categorias = np.array([ctx.nombres["categorias"][i] for i in ctx.cat_idx])
    departamentos = np.array([ctx.nombres["departamentos"][i] for i in ctx.dept_idx])
    salida: dict[str, dict[str, np.ndarray]] = {
        "total": {"total": valores.sum(axis=0)},
        "estado": {},
        "tienda": {},
        "categoria": {},
        "departamento": {},
        "categoria_tienda": {},
        "departamento_tienda": {},
    }
    for codigo in np.unique(estados):
        salida["estado"][str(codigo)] = valores[estados == codigo].sum(axis=0)
    for codigo in np.unique(tiendas):
        salida["tienda"][str(codigo)] = valores[tiendas == codigo].sum(axis=0)
    for codigo in np.unique(categorias):
        salida["categoria"][str(codigo)] = valores[categorias == codigo].sum(axis=0)
    for codigo in np.unique(departamentos):
        salida["departamento"][str(codigo)] = valores[departamentos == codigo].sum(axis=0)
    for categoria in np.unique(categorias):
        for tienda in np.unique(tiendas):
            mascara = (categorias == categoria) & (tiendas == tienda)
            if mascara.any():
                salida["categoria_tienda"][f"{categoria}|{tienda}"] = valores[mascara].sum(axis=0)
    for departamento in np.unique(departamentos):
        for tienda in np.unique(tiendas):
            mascara = (departamentos == departamento) & (tiendas == tienda)
            if mascara.any():
                salida["departamento_tienda"][f"{departamento}|{tienda}"] = valores[mascara].sum(axis=0)
    return salida


def firma_configuracion(modelo: dict, n_series: int | None = None) -> str:
    """Identificador reproducible de la configuración del modelo.

    Incluye el número de series para que una muestra técnica no comparta
    identificador con la ejecución del universo completo.
    """
    datos = {
        "variables": list(modelo["variables"]),
        "parametros": modelo["parametros"],
        "num_boost_round": modelo["num_boost_round"],
        "origenes": modelo["origenes_entrenamiento"],
        "fraccion_ceros": modelo["fraccion_ceros"],
        "series": n_series,
    }
    return hashlib.sha256(json.dumps(datos, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def crear_run_id(*, version: str, version_precios: str | None, variante: str, firma: str) -> str:
    """Identificador de ejecución (estable para la misma configuración y datos)."""
    firma_total = f"{version}|{version_precios or 'sin-precios'}|{variante}|{firma}"
    marca = hashlib.sha256(firma_total.encode("utf-8")).hexdigest()[:10]
    return f"run-{variante}-{marca}"


def memoria_mb() -> dict:  # pragma: no cover - depende del sistema
    """Memoria del proceso y disponible, si el sistema lo permite."""
    try:
        import resource  # type: ignore[import-not-found]

        pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        return {"pico_proceso_mb": round(pico, 1)}
    except Exception:  # noqa: BLE001 - Windows no tiene resource
        return {}


# --------------------------------------------------------------------------- #
# Agregación de métricas guardadas (para la API)
# --------------------------------------------------------------------------- #


def agregar_metricas_guardadas(lista: list[dict]) -> dict:
    """Agrupa métricas ya calculadas sumando errores absolutos y denominadores.

    ``mae * n`` es la suma de errores absolutos y ``sesgo * n`` la suma con signo; así
    el agregado no promedia porcentajes ni MAE de segmentos.
    """
    n = 0
    suma_abs = 0.0
    suma_error = 0.0
    denominador = 0.0
    for metrica in lista:
        if not metrica or not metrica.get("n"):
            continue
        n += int(metrica["n"])
        suma_abs += (metrica.get("mae") or 0.0) * metrica["n"]
        suma_error += (metrica.get("sesgo") or 0.0) * metrica["n"]
        denominador += metrica.get("denominador") or 0
    if n == 0:
        return {
            "n": 0,
            "mae": None,
            "sesgo": None,
            "wape": None,
            "denominador": 0,
            "denominador_cero": True,
            "ventanas": len(lista),
        }
    return {
        "n": n,
        "mae": round(suma_abs / n, 4),
        "sesgo": round(suma_error / n, 4),
        "wape": round(100 * suma_abs / denominador, 4) if denominador > 0 else None,
        "denominador": int(round(denominador)),
        "denominador_cero": denominador <= 0,
        "ventanas": len(lista),
    }


def agrupar_horizontes(evaluaciones: list[dict]) -> list[dict]:
    """Une ``por_horizonte`` de todos los orígenes en una serie de 1 a 28."""
    por_horizonte: dict[int, dict] = {}
    for evaluacion in evaluaciones:
        for punto in evaluacion.get("por_horizonte") or []:
            registro = por_horizonte.setdefault(
                int(punto["horizonte"]), {"candidato": [], "baseline": [], "fecha": punto.get("fecha")}
            )
            registro["candidato"].append(punto["candidato"])
            registro["baseline"].append(punto["baseline"])
    salida = []
    for horizonte in sorted(por_horizonte):
        registro = por_horizonte[horizonte]
        candidato = agregar_metricas_guardadas(registro["candidato"])
        baseline = agregar_metricas_guardadas(registro["baseline"])
        salida.append(
            {
                "horizonte": horizonte,
                "candidato": candidato,
                "baseline": baseline,
                "mejora_pct": _mejora(candidato["wape"], baseline["wape"]),
            }
        )
    return salida


def agrupar_segmentos(evaluaciones: list[dict]) -> list[dict]:
    """Une los segmentos de todos los orígenes por ``(nivel, clave)``."""
    agrupados: dict[tuple[str, str], dict] = {}
    for evaluacion in evaluaciones:
        for segmento in evaluacion.get("segmentos") or []:
            clave = (str(segmento["nivel"]), str(segmento["clave"]))
            registro = agrupados.setdefault(
                clave, {"series": 0, "candidato": [], "baseline": [], "agregado_c": [], "agregado_b": []}
            )
            registro["series"] = max(registro["series"], int(segmento.get("series") or 0))
            registro["candidato"].append(segmento["candidato"])
            registro["baseline"].append(segmento["baseline"])
            agregado = segmento.get("agregado") or {}
            if agregado:
                registro["agregado_c"].append(agregado.get("candidato") or {})
                registro["agregado_b"].append(agregado.get("baseline") or {})
    salida = []
    for (nivel, clave), registro in sorted(agrupados.items()):
        candidato = agregar_metricas_guardadas(registro["candidato"])
        baseline = agregar_metricas_guardadas(registro["baseline"])
        agregado = None
        if registro["agregado_c"]:
            cand_agregado = agregar_metricas_guardadas(registro["agregado_c"])
            base_agregado = agregar_metricas_guardadas(registro["agregado_b"])
            agregado = {
                "candidato": cand_agregado,
                "baseline": base_agregado,
                "mejora_pct": _mejora(cand_agregado["wape"], base_agregado["wape"]),
            }
        salida.append(
            {
                "nivel": nivel,
                "clave": clave,
                "series": registro["series"],
                "candidato": candidato,
                "baseline": baseline,
                "mejora_pct": _mejora(candidato["wape"], baseline["wape"]),
                "agregado": agregado,
            }
        )
    return salida
