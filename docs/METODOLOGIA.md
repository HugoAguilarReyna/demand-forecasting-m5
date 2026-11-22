# Metodología del pronóstico a 28 días

## El desafío

M5 (Walmart, Kaggle) pide pronosticar **28 días** de ventas diarias por
combinación de tienda, departamento y producto. Este repositorio contiene el
pipeline de un enfoque global: un solo modelo condicionado al horizonte, en lugar
de un modelo por serie.

La constante vive en un único lugar:

```python
# lib/app/comercial/fuentes.py
DIAS_HORIZONTE = 28
```

`modelo.py` la hereda (`HORIZONTE = fuentes.DIAS_HORIZONTE`) y una prueba del
repositorio falla si alguien la redefine o la cambia en silencio.

## Modelo

| Decisión | Elección |
|---|---|
| Familia | **LightGBM** global, condicionado al horizonte |
| Objetivo | Se comparó con `tweedie` en una muestra y se conservó el de mejor desempeño |
| Referencia | Baseline estacional semanal |
| Variables | Rezagos (`lag_*`), medias y desviaciones sobre ventanas (`media_*`, `std_*`) y medias por día de la semana (`dow_media_*`) |
| Precio | Para los 28 días pronosticados se usa el **último precio conocido** de la semana |
| Recorte | Las predicciones negativas se recortan a **cero**, también al evaluar |

El modelo es único y global: no hay un bosque por SKU ni reentrenamiento por
serie durante la evaluación.

## Protocolo de evaluación

El entrenamiento separa dos fases, y esa separación es el núcleo del método:

1. **Desarrollo (`--variantes`)** — se entrenan las variantes candidatas sobre
   *cortes de desarrollo* y se comparan **por WAPE**.
2. **Final** — se evalúa el modelo elegido en cortes posteriores y se genera el
   pronóstico. Según la ficha publicada, esos cortes se consultaron para confirmar
   la publicación: no constituyen un holdout intacto. Para una evaluación nueva
   hay que reservar otro periodo o conjunto antes de tomar decisiones.

Métricas reportadas por horizonte y agregadas:

| Métrica | Qué mide |
|---|---|
| **WAPE** | Error absoluto ponderado por el volumen; es la métrica de selección. |
| **MAE** | Error absoluto medio en unidades. |
| **Sesgo** | Suma de errores con signo; detecta sobre/subpronóstico sistemático. |
| **Ventanas** | Número de ventanas evaluadas que respaldan cada cifra. |

Se evalúa **por horizonte** (día 1 a 28) porque el error no es estable dentro de
la ventana: un solo número agregado ocultaría el deterioro del tramo largo.

## Backtesting a 28 días

Cada corte define un origen y pronostica los 28 días siguientes. El bucle salta
de ventana en ventana; si una ventana no cabe en el calendario disponible se
descarta en lugar de truncarse (el código lo verifica antes de pronosticar). Así,
todas las cifras publicadas provienen de ventanas completas de 28 días.

El pronóstico final se apila sobre el último origen disponible y se acompaña de
las métricas de los cortes reservados; no se mezcla con las cifras de desarrollo.

## Qué se publica y qué no

Se publican los scripts, el paquete de modelado y las métricas en MongoDB
(`M5_EVALUACION`, `M5_PRONOSTICOS`, `M5_AGREGADOS`). No se publican los CSV de M5
ni los agregados derivados: ver [`DATOS.md`](DATOS.md).

## Límites

- El WAPE de un corte no es una promesa de desempeño en otro periodo.
- La selección por WAPE favorece el volumen: series pequeñas pesan poco aunque su
  error relativo sea alto.
- El baseline semanal es exigente en series con estacionalidad fuerte; superarlo
  no implica haber capturado estructura causal.
- El recorte a cero mejora el error cuando el modelo predice negativo, pero
  enmascara el problema de especificación.
