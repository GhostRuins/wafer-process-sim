from __future__ import annotations

from wafer_sim.fdc.spc_engine import SPCAlert, SPCEngine


class FDCFeatureAugmentor:
    def __init__(self, window_size: int = 25) -> None:
        self.spc = SPCEngine(window_size=window_size)
        self.last_alerts: list[SPCAlert] = []

    def augment(self, feature_dict: dict[str, float]) -> dict[str, float]:
        base = {k: float(v) for k, v in feature_dict.items()}
        self.last_alerts = self.spc.update(base)
        spc_feats = self.spc.get_alert_features(base)
        return {**base, **spc_feats}

    @property
    def last_summary(self) -> dict[str, float | bool]:
        return self.spc.last_summary
