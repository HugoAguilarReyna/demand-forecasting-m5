"""Datos **sintéticos** para las pruebas del Atlas comercial.

Advertencia: estos archivos **no son M5**. Se generan con un patrón determinista
(estacionalidad semanal + tendencia suave + algunos ceros) y existen solo para
probar ingesta, reconciliación, filtros, concentración y baseline sin depender de
la descarga de Kaggle. El tablero público nunca los usa: su estado es «datos
pendientes de importar» mientras no estén los CSV reales.

El generador escribe CSV con el mismo esquema de M5 (``id``, ``item_id``,
``dept_id``, ``cat_id``, ``store_id``, ``state_id`` y ``d_1…d_N``) y un
``calendar.csv`` con las columnas que consume el importador.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

DIAS = 70
INICIO = date(2011, 1, 29)
TIENDAS = (("CA_1", "CA"), ("TX_1", "TX"), ("WI_1", "WI"))
PRODUCTOS = (
    ("FOODS_1_001", "FOODS_1", "FOODS"),
    ("FOODS_1_002", "FOODS_1", "FOODS"),
    ("HOBBIES_1_001", "HOBBIES_1", "HOBBIES"),
)


def _serie(semilla: int, dias: int) -> list[int]:
    """Serie determinista con estacionalidad semanal y ceros puntuales."""
    salida: list[int] = []
    for dia in range(dias):
        base = 10 + (semilla % 5) * 3
        estacional = (dia + semilla) % 7
        valor = base + estacional * 2 + (dia // 14)
        if (dia + semilla) % 23 == 0:
            valor = 0
        salida.append(int(valor))
    return salida


def escribir(carpeta: Path, *, inicio: date | None = None, dias: int | None = None) -> dict:
    """Escribe ``sales_train_evaluation.csv`` y ``calendar.csv`` sintéticos.

    ``inicio`` y ``dias`` permiten generar un rango largo (por ejemplo, varios
    años completos) para probar la comparación histórica sin descargar M5.
    """
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    inicio = inicio or INICIO
    dias = dias or DIAS
    ventas = carpeta / "sales_train_evaluation.csv"
    calendario = carpeta / "calendar.csv"

    with calendario.open("w", encoding="utf-8", newline="") as manejador:
        escritor = csv.writer(manejador)
        escritor.writerow(
            ["date", "wm_yr_wk", "weekday", "wday", "month", "year", "d",
             "event_name_1", "event_type_1", "event_name_2", "event_type_2",
             "snap_CA", "snap_TX", "snap_WI"]
        )
        for dia in range(1, dias + 1):
            fecha = inicio + timedelta(days=dia - 1)
            escritor.writerow(
                [fecha.isoformat(), 11100 + dia // 7, fecha.strftime("%A"), fecha.isoweekday(),
                 fecha.month, fecha.year, f"d_{dia}", "", "", "", "",
                 1 if fecha.day % 10 == 0 else 0, 0, 0]
            )

    with ventas.open("w", encoding="utf-8", newline="") as manejador:
        escritor = csv.writer(manejador)
        escritor.writerow(
            ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
            + [f"d_{dia}" for dia in range(1, dias + 1)]
        )
        semilla = 0
        for store_id, state_id in TIENDAS:
            for item_id, dept_id, cat_id in PRODUCTOS:
                semilla += 1
                serie = _serie(semilla, dias)
                if store_id == "WI_1" and cat_id == "HOBBIES":
                    serie = [0] * dias  # sirve para probar total cero
                escritor.writerow(
                    [f"{item_id}_{store_id}", item_id, dept_id, cat_id, store_id, state_id]
                    + serie
                )
    return {"carpeta": str(carpeta), "dias": dias, "series": len(TIENDAS) * len(PRODUCTOS)}


def total_esperado(dias: int | None = None) -> int:
    """Suma de todas las series escritas (para reconciliar en las pruebas)."""
    dias = dias or DIAS
    total = 0
    semilla = 0
    for store_id, _ in TIENDAS:
        for _, _, cat_id in PRODUCTOS:
            semilla += 1
            serie = [0] * dias if (store_id == "WI_1" and cat_id == "HOBBIES") else _serie(semilla, dias)
            total += sum(serie)
    return total
