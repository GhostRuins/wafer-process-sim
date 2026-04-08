"""
Data models for CVD wafer process simulation.

All physical quantities use explicit units in field names and docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ProcessParams:
    """Process parameters for a single deposition run.

    Parameters
    ----------
    temperature_c : float
        Chamber temperature in degrees Celsius (°C).
    pressure_mtorr : float
        Chamber pressure in millitorr (mTorr).
    gas_flow_sccm : float
        Total precursor / carrier gas flow in standard cubic centimeters per
        minute (sccm).
    rf_power_w : float
        RF plasma power in watts (W).
    deposition_time_s : float
        Deposition duration in seconds (s).
    tool_id : int
        Categorical tool identifier; must be 0, 1, or 2.
    wafer_position_x : float, optional
        Normalized wafer horizontal coordinate in [-1, 1] (unitless).
    wafer_position_y : float, optional
        Normalized wafer vertical coordinate in [-1, 1] (unitless).
    """

    temperature_c: float
    pressure_mtorr: float
    gas_flow_sccm: float
    rf_power_w: float
    deposition_time_s: float
    tool_id: int
    wafer_position_x: float = 0.0
    wafer_position_y: float = 0.0


@dataclass
class YieldModelConfig:
    """Configuration for yield models and wafer-level statistics.

    Parameters
    ----------
    yield_model : {'murphy', 'seeds', 'negative_binomial'}
        Yield aggregation model. ``murphy`` uses the clustered-defect Murphy
        form; ``seeds`` is the Poisson (Seeds) limit; ``negative_binomial``
        uses the gamma–Poisson closed form.
    a_die_cm2 : float
        Die area in square centimeters (cm²) used in defect-limited yield.
    grid_nx : int
        Number of die columns across the wafer for uniformity statistics.
    grid_ny : int
        Number of die rows across the wafer for uniformity statistics.
    k_u : float, optional
        Sensitivity of yield to thickness coefficient of variation
        ``sigma_t / mu_t`` (unitless). If None, a value from the physics engine
        module state is used (typically 0.5–1.0).
    """

    yield_model: Literal["murphy", "seeds", "negative_binomial"] = "murphy"
    a_die_cm2: float = 1.0
    grid_nx: int = 12
    grid_ny: int = 12
    k_u: Optional[float] = None


@dataclass
class DieResult:
    """Point- or die-level simulation outcomes.

    Parameters
    ----------
    film_thickness_nm : float
        Local film thickness in nanometers (nm).
    defect_density_cm2 : float
        Local defect density in defects per square centimeter (cm⁻²).
    yield_ : float
        Local yield estimate in [0, 1] (unitless). For a single location,
        thickness uniformity penalty is not applied (no wafer-level
        ``sigma_t``); only the selected defect-limited yield model is used.
    """

    film_thickness_nm: float
    defect_density_cm2: float
    yield_: float


@dataclass
class WaferResult:
    """Full-wafer aggregate simulation outcomes.

    Parameters
    ----------
    mean_film_thickness_nm : float
        Mean film thickness over the die grid in nanometers (nm).
    std_film_thickness_nm : float
        Standard deviation of film thickness over the die grid (nm).
    mean_defect_density_cm2 : float
        Mean defect density over the die grid (cm⁻²).
    yield_ : float
        Wafer-level yield in [0, 1] (unitless), including Murphy / Seeds /
        negative-binomial defect statistics and multiplicative thickness
        uniformity factor.
    """

    mean_film_thickness_nm: float
    std_film_thickness_nm: float
    mean_defect_density_cm2: float
    yield_: float


@dataclass
class _Bounds:
    """Internal: validated numeric ranges for ProcessParams."""

    t_min_c: float = 150.0
    t_max_c: float = 650.0
    p_min_mtorr: float = 1.0
    p_max_mtorr: float = 5000.0
    flow_min_sccm: float = 1.0
    flow_max_sccm: float = 500.0
    rf_min_w: float = 10.0
    rf_max_w: float = 1500.0
    time_min_s: float = 1.0
    time_max_s: float = 7200.0


def validate_process_params(params: ProcessParams, bounds: Optional[_Bounds] = None) -> None:
    """Validate ``ProcessParams`` against realistic CVD bounds.

    Raises
    ------
    ValueError
        If any field is outside allowed bounds. Messages include units.

    Parameters
    ----------
    params : ProcessParams
        Parameters to validate.
    bounds : _Bounds, optional
        Override default bounds (for testing).
    """
    b = bounds or _Bounds()
    if params.tool_id not in (0, 1, 2):
        raise ValueError("tool_id must be 0, 1, or 2 (unitless categorical index).")
    if not (b.t_min_c <= params.temperature_c <= b.t_max_c):
        raise ValueError(
            f"temperature_c must be in [{b.t_min_c}, {b.t_max_c}] °C, got {params.temperature_c} °C."
        )
    if not (b.p_min_mtorr <= params.pressure_mtorr <= b.p_max_mtorr):
        raise ValueError(
            f"pressure_mtorr must be in [{b.p_min_mtorr}, {b.p_max_mtorr}] mTorr, "
            f"got {params.pressure_mtorr} mTorr."
        )
    if not (b.flow_min_sccm <= params.gas_flow_sccm <= b.flow_max_sccm):
        raise ValueError(
            f"gas_flow_sccm must be in [{b.flow_min_sccm}, {b.flow_max_sccm}] sccm, "
            f"got {params.gas_flow_sccm} sccm."
        )
    if not (b.rf_min_w <= params.rf_power_w <= b.rf_max_w):
        raise ValueError(
            f"rf_power_w must be in [{b.rf_min_w}, {b.rf_max_w}] W, got {params.rf_power_w} W."
        )
    if not (b.time_min_s <= params.deposition_time_s <= b.time_max_s):
        raise ValueError(
            f"deposition_time_s must be in [{b.time_min_s}, {b.time_max_s}] s, "
            f"got {params.deposition_time_s} s."
        )
    if not (-1.0 <= params.wafer_position_x <= 1.0 and -1.0 <= params.wafer_position_y <= 1.0):
        raise ValueError(
            "wafer_position_x and wafer_position_y must lie in [-1, 1] (normalized coordinates)."
        )
