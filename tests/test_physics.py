"""Tests for wafer_sim.physics."""

from __future__ import annotations

import numpy as np

from wafer_sim.models import ProcessParams, YieldModelConfig
from wafer_sim.physics import (
    RF_NOMINAL_W,
    _get_state,
    _murphy_yield,
    _seeds_yield,
    defect_density_cm2,
    film_thickness_nm,
    generate_die,
    generate_wafer,
    init_physics,
    uniformity_factor,
)


def _nominal_params(**kwargs) -> ProcessParams:
    base = dict(
        temperature_c=400.0,
        pressure_mtorr=500.0,
        gas_flow_sccm=50.0,
        rf_power_w=RF_NOMINAL_W,
        deposition_time_s=60.0,
        tool_id=0,
        wafer_position_x=0.0,
        wafer_position_y=0.0,
    )
    base.update(kwargs)
    return ProcessParams(**base)


def test_thickness_decreases_with_lower_temperature():
    init_physics(7)
    state = _get_state()
    t_high = film_thickness_nm(450.0, 500.0, 50.0, 60.0, 0, 0.0, 0.0, state)
    t_low = film_thickness_nm(300.0, 500.0, 50.0, 60.0, 0, 0.0, 0.0, state)
    assert t_low < t_high, "Arrhenius term should reduce thickness at lower temperature (nm)."


def test_thickness_nonuniformity_within_ten_percent_of_baseline():
    init_physics(11)
    state = _get_state()
    p = _nominal_params()
    base = film_thickness_nm(
        p.temperature_c,
        p.pressure_mtorr,
        p.gas_flow_sccm,
        p.deposition_time_s,
        p.tool_id,
        0.0,
        0.0,
        state,
    )
    for x, y in [(0.9, 0.0), (-0.6, 0.5), (0.2, -0.85)]:
        if x * x + y * y > 1.0:
            continue
        t = film_thickness_nm(
            p.temperature_c,
            p.pressure_mtorr,
            p.gas_flow_sccm,
            p.deposition_time_s,
            p.tool_id,
            x,
            y,
            state,
        )
        rel = abs(t / base - 1.0)
        assert rel <= 0.105, f"Thickness multiplicative factor should stay within ±10% (got {rel:.4f})."


def test_yield_in_zero_one_for_random_combinations():
    init_physics(13)
    rng = np.random.default_rng(21)
    for _ in range(40):
        p = ProcessParams(
            temperature_c=float(rng.uniform(250.0, 550.0)),
            pressure_mtorr=float(rng.uniform(50.0, 2000.0)),
            gas_flow_sccm=float(rng.uniform(10.0, 120.0)),
            rf_power_w=float(rng.uniform(50.0, 800.0)),
            deposition_time_s=float(rng.uniform(30.0, 600.0)),
            tool_id=int(rng.integers(0, 3)),
        )
        seed = int(rng.integers(0, 10_000_000))
        w = generate_wafer(p, seed, YieldModelConfig(grid_nx=8, grid_ny=8))
        assert 0.0 <= w.yield_ <= 1.0
        d = generate_die(p, 0.1, -0.2, seed + 1)
        assert 0.0 <= d.yield_ <= 1.0


def test_defect_density_increases_with_pressure_power_law():
    init_physics(19)
    state = _get_state()
    p_lo = _nominal_params(pressure_mtorr=100.0)
    p_hi = _nominal_params(pressure_mtorr=800.0)
    empty = np.array([], dtype=np.int64)
    d_lo = defect_density_cm2(p_lo, 0.2, -0.1, state, None, empty)
    d_hi = defect_density_cm2(p_hi, 0.2, -0.1, state, None, empty)
    assert d_hi > d_lo, "Defect density should increase with pressure via power-law base term (cm⁻²)."


def test_rf_u_shaped_defect_density():
    init_physics(23)
    state = _get_state()
    p_nom = _nominal_params(rf_power_w=RF_NOMINAL_W)
    p_low = _nominal_params(rf_power_w=80.0)
    p_high = _nominal_params(rf_power_w=900.0)
    empty = np.array([], dtype=np.int64)
    d_nom = defect_density_cm2(p_nom, 0.0, 0.0, state, None, empty)
    d_low = defect_density_cm2(p_low, 0.0, 0.0, state, None, empty)
    d_high = defect_density_cm2(p_high, 0.0, 0.0, state, None, empty)
    assert d_low > d_nom and d_high > d_nom, "RF defect term should be U-shaped around nominal (W)."


def test_murphy_more_pessimistic_than_seeds_at_moderate_da_with_exponent():
    """User Murphy form uses an exponent; for moderate D*A, exponent must exceed ~2.2 to be < Seeds."""
    da = 1.0
    seeds = _seeds_yield(da)
    murphy_lo = _murphy_yield(da, 1.0)
    murphy_hi = _murphy_yield(da, 3.0)
    assert murphy_hi < seeds < murphy_lo, (
        "At D*A=1, inner^3 is below Seeds while inner^1 is above (clustering pessimism check)."
    )


def test_murphy_exponent_increases_pessimism():
    da = 0.8
    y1 = _murphy_yield(da, 1.0)
    y2 = _murphy_yield(da, 2.0)
    assert y2 < y1, "Larger Murphy exponent should reduce yield for fixed D*A (unitless)."


def test_uniformity_factor_non_negative():
    for sigma in np.linspace(0.0, 50.0, 20):
        for mu in np.linspace(10.0, 200.0, 20):
            for k_u in np.linspace(0.5, 1.0, 6):
                u = uniformity_factor(sigma, mu, k_u)
                assert u >= 0.0
                assert u <= 1.0
