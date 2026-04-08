"""Map API models to wafer_sim domain objects."""

from __future__ import annotations

from api.schemas import ProcessParams as APIProcessParams
from api.util import tool_name_to_int
from wafer_sim.models import ProcessParams as SimProcessParams


def api_to_sim_params(p: APIProcessParams) -> SimProcessParams:
    return SimProcessParams(
        temperature_c=p.temperature,
        pressure_mtorr=p.pressure,
        gas_flow_sccm=p.gas_flow,
        rf_power_w=p.rf_power,
        deposition_time_s=p.deposition_time,
        tool_id=tool_name_to_int(p.tool_id),
    )
