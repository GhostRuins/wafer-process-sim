"""ML yield prediction endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.convert import api_to_sim_params
from api.deps import AppStateDep
from api.schemas import ProcessParams, YieldPrediction

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("", response_model=YieldPrediction, summary="Predict wafer yield from process parameters")
async def predict(body: ProcessParams, state: AppStateDep) -> YieldPrediction:
    if state.ensemble is None:
        raise HTTPException(
            status_code=503,
            detail="Trained yield_ensemble.joblib not found; train models or set ML_ARTIFACTS_DIR.",
        )
    from ml.yield_model import predict_yield

    sim_p = api_to_sim_params(body)
    out = predict_yield(sim_p, ensemble=state.ensemble, return_shap=True)
    flags = [f"{f.parameter}: {f.message}" for f in out.risk_flags]
    return YieldPrediction(
        predicted_yield=out.predicted_yield,
        confidence_interval=out.confidence_interval,
        shap_breakdown=out.shap_breakdown or {},
        risk_flags=flags,
        spc_alerts_active=out.spc_alerts_active,
        spc_severity=out.spc_severity,
        spc_alert_count=out.spc_alert_count,
    )
