"""AVISO · Solo pruebas: importa un conjunto SINTÉTICO en la base local del proyecto.

No es M5 y no debe publicarse. Sirve para verificar la interfaz y el flujo completo
cuando los CSV reales no están disponibles; deja el estado explícitamente marcado
como sintético y se retira con ``--limpiar``.

Uso:
    python -m scripts.demo_sintetica_m5            # importa y calcula el baseline
    python -m scripts.demo_sintetica_m5 --limpiar  # borra la versión de prueba

Origen del texto original:

No forma parte del flujo público. Sirve para verificar la interfaz cuando los CSV
reales de M5 no están disponibles; el estado que deja es explícitamente sintético y
se retira con `--limpiar` (o con scripts/importar_m5 --estado tras borrar).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from app.comercial import forecast, ingesta
from app.config import get_settings
from app.db.mongo import init_database
from pipeline import fixture_sintetico as m5_fixture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Importa el conjunto sintético de pruebas.")
    parser.add_argument("--limpiar", action="store_true", help="Borra la versión publicada de prueba.")
    parser.add_argument(
        "--si-entiendo-que-es-sintetico",
        action="store_true",
        help="Obligatorio: confirma que se cargan datos de prueba y no M5.",
    )
    args = parser.parse_args(argv)

    if not args.limpiar and not getattr(args, "si_entiendo_que_es_sintetico", False):
        print(
            "Este comando carga datos SINTÉTICOS de prueba, no M5.\n"
            "Si es lo que quieres, repite con --si-entiendo-que-es-sintetico.\n"
            "Para la carga real: coloca los CSV en backend/data/m5 y ejecuta\n"
            "python -m scripts.importar_m5"
        )
        return 2

    base = init_database(get_settings())
    if args.limpiar:
        for version in [v for v in base.m5_db["m5_importaciones"].distinct("_id")]:
            ingesta.limpiar_version(base.m5_db, version)
        base.m5_db["m5_estado"].delete_many({})
        base.m5_db["m5_calendario"].delete_many({})
        print(json.dumps(ingesta.estado_publico(base.m5_db), ensure_ascii=False, indent=2))
        return 0

    carpeta = Path(tempfile.mkdtemp(prefix="m5-sintetico-"))
    m5_fixture.escribir(carpeta)
    manifiesto = ingesta.importar(base.m5_db, carpeta)
    resultado = forecast.calcular(base.m5_db, manifiesto["version"], ventanas=2)
    print(
        json.dumps(
            {
                "version": manifiesto["version"],
                "origen": "SINTÉTICO (pruebas, no es M5)",
                "cobertura": manifiesto["cobertura"],
                "conteos": manifiesto["conteos"],
                "validaciones": manifiesto["validaciones"],
                "baseline": resultado,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
