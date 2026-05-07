from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class SPCAlert:
    param: str
    rule: str
    severity: int
    value: float
    ucl: float
    lcl: float


class SPCEngine:
    def __init__(self, window_size: int = 25) -> None:
        self.window_size = max(3, int(window_size))
        self._history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.window_size))
        self._active_alerts: dict[str, list[SPCAlert]] = defaultdict(list)
        self._last_params: dict[str, float] = {}

    def update(self, params: dict[str, float]) -> list[SPCAlert]:
        alerts: list[SPCAlert] = []
        self._last_params = dict(params)
        for name, raw in params.items():
            val = float(raw)
            hist = self._history[name]
            hist.append(val)
            arr = list(hist)
            if len(arr) < 3:
                self._active_alerts[name] = []
                continue

            mean = sum(arr) / len(arr)
            var = sum((x - mean) ** 2 for x in arr) / len(arr)
            sigma = max(var**0.5, 1e-9)
            ucl = mean + 3.0 * sigma
            lcl = mean - 3.0 * sigma

            param_alerts: list[SPCAlert] = []

            # Rule 1: one point beyond 3 sigma from center line.
            if abs(val - mean) > 3.0 * sigma:
                param_alerts.append(SPCAlert(name, "rule_1", 2, val, ucl, lcl))

            # Rule 2: 9 consecutive points on same side of mean.
            if len(arr) >= 9:
                last = arr[-9:]
                if all(x > mean for x in last) or all(x < mean for x in last):
                    param_alerts.append(SPCAlert(name, "rule_2", 1, val, ucl, lcl))

            # Rule 3: 6 consecutive monotone trend.
            if len(arr) >= 6:
                last = arr[-6:]
                if all(last[i] < last[i + 1] for i in range(5)) or all(
                    last[i] > last[i + 1] for i in range(5)
                ):
                    param_alerts.append(SPCAlert(name, "rule_3", 1, val, ucl, lcl))

            # Rule 4: two of three consecutive points beyond 2 sigma on same side.
            if len(arr) >= 3:
                last = arr[-3:]
                hi = sum(x > mean + 2.0 * sigma for x in last)
                lo = sum(x < mean - 2.0 * sigma for x in last)
                if hi >= 2 or lo >= 2:
                    param_alerts.append(SPCAlert(name, "rule_4", 1, val, ucl, lcl))

            self._active_alerts[name] = param_alerts
            alerts.extend(param_alerts)

        return alerts

    def get_alert_features(self, params: dict[str, float]) -> dict[str, float]:
        features: dict[str, float] = {}
        active_count = 0
        severity_sum = 0.0
        n_params = max(len(params), 1)

        for name in params:
            alerts = self._active_alerts.get(name, [])
            is_active = 1.0 if len(alerts) > 0 else 0.0
            features[f"spc_alert_{name}"] = is_active
            active_count += int(is_active)
            severity_sum += float(sum(a.severity for a in alerts))

        features["spc_alert_count"] = float(active_count)
        features["spc_severity_score"] = float(severity_sum / n_params)
        return features

    @property
    def last_summary(self) -> dict[str, float | bool]:
        if not self._last_params:
            return {
                "spc_alerts_active": False,
                "spc_alert_count": 0.0,
                "spc_severity": 0.0,
            }
        feats = self.get_alert_features(self._last_params)
        sev = float(feats.get("spc_severity_score", 0.0))
        if not isfinite(sev):
            sev = 0.0
        count = float(feats.get("spc_alert_count", 0.0))
        return {
            "spc_alerts_active": count > 0.0,
            "spc_alert_count": count,
            "spc_severity": sev,
        }
