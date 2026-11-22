"""Entrena el motor real en memoria con datos sintéticos; no requiere MongoDB.

python -m pipeline.verificar_modelo --salida research/smoke.json
El resultado comprueba ejecución y trazabilidad; no reproduce el benchmark M5.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np
from app.comercial import modelo

SEED = 20261121

def contexto() -> modelo.Contexto:
    rng = np.random.default_rng(SEED)
    n, days = 36, 600
    t = np.arange(days)
    rates = rng.uniform(2, 15, (n, 1)) * (1 + .3*np.sin(2*np.pi*t/7))[None, :]
    sales = rng.poisson(rates).astype(np.int32)
    sales[rng.random(sales.shape) < .08] = 0
    cal = {
        'd': t+1, 'dow': t%7+1, 'month': t//30%12+1, 'day': t%28+1,
        'doy': t%365+1, 'semana_anio': t//7%52+1, 'fin_de_semana': (t%7>=5).astype(int),
        'desde_inicio': t,
    }
    for name in ('snap_ca','snap_tx','snap_wi','evento_1','evento_2','tipo_1','tipo_2'):
        cal[name] = np.zeros(days, dtype=np.int32)
    return modelo.Contexto(
        version='synthetic-smoke-v1', version_precios=None,
        series=[(f'item_{i}',f'store_{i%3}') for i in range(n)],
        item_idx=np.arange(n), store_idx=np.arange(n)%3,
        cat_idx=np.arange(n)%2, dept_idx=np.arange(n)%2, state_idx=np.arange(n)%3,
        ventas=sales, calendario=cal, semana_por_dia=t//7, semanas=list(range(86)),
        precios_serie=None, precios_rellenados=None,
    )

def ejecutar() -> dict:
    ctx = contexto()
    origin = 540
    before, names = modelo.construir_matriz(ctx, origin, con_precios=False)
    changed = contexto()
    changed.ventas[:, origin:] = 99999
    after, _ = modelo.construir_matriz(changed, origin, con_precios=False)
    if not np.array_equal(before, after, equal_nan=True):
        raise AssertionError('Las variables usan ventas futuras')
    trained = modelo.entrenar(ctx, [420,448,476], con_precios=False,
                             semilla=SEED, num_boost_round=30, fraccion_ceros=1)
    baseline = modelo.predecir_baseline(ctx, origin)
    prediction, coverage = modelo.predecir(ctx, trained, origin, baseline=baseline)
    truth = ctx.ventas[:, origin:origin+28]
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise AssertionError('Predicción incompleta o no finita')
    if min(trained['origenes_entrenamiento']) < modelo.MINIMO_HISTORIA or max(trained['origenes_entrenamiento'])+28 > origin:
        raise AssertionError('Las etiquetas de entrenamiento alcanzan la evaluación')
    return {
        'tipo': 'prueba-sintetica-de-ejecucion-no-benchmark-M5', 'semilla': SEED,
        'dataset_sha256': hashlib.sha256(ctx.ventas.tobytes()).hexdigest(),
        'variables': names, 'origenes_entrenamiento': trained['origenes_entrenamiento'],
        'origen_evaluacion': origin, 'horizonte': 28, 'series': ctx.n_series,
        'modelo': modelo.metricas(prediction-truth, truth),
        'baseline': modelo.metricas(baseline-truth, truth),
        'por_horizonte': [{'dia': h+1, 'modelo': modelo.metricas(prediction[:,h]-truth[:,h],truth[:,h]),
                          'baseline': modelo.metricas(baseline[:,h]-truth[:,h],truth[:,h])} for h in range(28)],
        'cobertura': coverage, 'parametros': trained['parametros'],
        'rondas': trained['num_boost_round'], 'sin_ventas_futuras': True,
        'entorno': {'python': platform.python_version(), **{p: importlib.metadata.version(p) for p in ('numpy','lightgbm','pymongo')}},
        'limitacion': 'Datos sintéticos conocidos durante el desarrollo. No es un holdout de M5 ni evidencia de generalización comercial.',
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--salida', type=Path, default=Path('research/smoke.json'))
    args = parser.parse_args()
    report = ejecutar()
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('tipo','modelo','baseline','sin_ventas_futuras')}, ensure_ascii=False))

if __name__ == '__main__':
    main()
