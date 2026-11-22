# Datos, licencias y estado de ejecución

## Conjunto de origen

**M5 Forecasting — Accuracy** (Walmart, Kaggle): ventas diarias por tienda,
departamento y producto, calendario y precios, con horizonte de evaluación de
28 días.

| Aspecto | Estado en este repositorio |
|---|---|
| CSV de origen | **No se redistribuyen.** Se descargan del concurso y se importan localmente. |
| Agregados derivados | No versionados; los produce el pipeline. |
| Código del pipeline | Sí, completo. |
| Métricas publicadas | En MongoDB durante la ejecución; no se versionan. |

La descarga se realiza con la cuenta del concurso y queda sujeta a sus términos.
No se incluyen identificadores de clientes: M5 no contiene datos personales.

## Estado de ejecución

El pipeline incluye su configuración y adaptador local. Después de `pip install -e .`,
la ejecución sintética en memoria funciona sin datos externos ni MongoDB.
La ejecución completa sobre M5 requiere:

1. **Los CSV de M5** descargados e importados con `python -m pipeline.importar_m5`.
2. **Una instancia de MongoDB** accesible; las colecciones de destino son
   `M5_EVALUACION`, `M5_PRONOSTICOS` y `M5_AGREGADOS`.
3. **Python 3.11+**: el motor funciona también en Windows; la medición opcional de memoria con `resource` se omite si no está disponible.

CI verifica los contratos estáticos y entrena el motor real con un conjunto sintético.
El informe conserva la configuración y las métricas; no reproduce las cifras M5.

Esta honestidad es deliberada: un repositorio que promete ejecutarse y falla al
primer `pip install` cuesta más que uno que declara sus requisitos.

## Unidades

- **Ventas**: unidades vendidas por día; la suma de los 28 días es el objetivo.
- **Precio**: el último precio conocido de la semana para toda la ventana de
  pronóstico.
- **WAPE**: error absoluto ponderado por volumen, agregado sobre las ventanas
  evaluadas.
- **Sesgo**: suma de errores con signo; no es un porcentaje.

## Atribución

> M5 Forecasting — Accuracy, Walmart, Kaggle. Los datos se usan bajo los términos
> del concurso y no se redistribuyen.

Este repositorio no incluye un archivo `LICENSE` propio: se respetan las
condiciones de las fuentes citadas.
