"""Pruebas de comportamiento sobre el motor importable, sin MongoDB."""
import numpy as np
import pytest
from app.comercial import modelo
from app.comercial.conexion import DestinoNoPermitido, resolver_destino
from pipeline.verificar_modelo import contexto

def test_cambiar_el_futuro_no_cambia_variables():
    ctx = contexto()
    original, _ = modelo.construir_matriz(ctx, 540, con_precios=False)
    ctx.ventas[:, 540:] = 999999
    altered, _ = modelo.construir_matriz(ctx, 540, con_precios=False)
    np.testing.assert_array_equal(original, altered)

def test_baseline_repite_exclusivamente_la_ultima_semana():
    ctx = contexto()
    result = modelo.predecir_baseline(ctx, 540)
    np.testing.assert_array_equal(result, np.tile(ctx.ventas[:,533:540], (1,4)))

def test_metricas_no_inventan_porcentaje_con_demanda_cero():
    result = modelo.metricas(np.array([1.,-1.]), np.zeros(2))
    assert result['wape'] is None
    assert result['mae'] == 1
    assert result['sesgo'] == 0

def test_calendario_incompleto_se_rechaza():
    ctx = contexto()
    with pytest.raises(ValueError, match='calendario'):
        modelo.construir_matriz(ctx, 580, con_precios=False)

def test_conexion_no_admite_atlas_ni_destinos_remotos():
    for uri in ['mongodb+srv://example.invalid/test', 'mongodb://example.invalid/test']:
        with pytest.raises(DestinoNoPermitido):
            resolver_destino(uri)
