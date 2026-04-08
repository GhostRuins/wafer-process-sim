"""HTTP tests for the FastAPI service (async httpx client)."""

from __future__ import annotations

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from api.main import app


@pytest_asyncio.fixture
async def client():
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_simulate_returns_yield_and_maps(client: AsyncClient):
    payload = {
        "temperature": 400.0,
        "pressure": 50.0,
        "gas_flow": 80.0,
        "rf_power": 200.0,
        "deposition_time": 120.0,
        "tool_id": "tool_A",
    }
    r = await client.post("/api/simulate", params={"seed": 1}, json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "yield" in data
    assert 0.0 <= data["yield"] <= 1.0
    assert "thickness_map" in data and len(data["thickness_map"]) > 0
    assert "dies" in data and len(data["dies"]) > 0


@pytest.mark.asyncio
async def test_predict_without_model_is_503(client: AsyncClient):
    payload = {
        "temperature": 400.0,
        "pressure": 50.0,
        "gas_flow": 80.0,
        "rf_power": 200.0,
        "deposition_time": 120.0,
        "tool_id": "tool_A",
    }
    r = await client.post("/api/predict", json=payload)
    if r.status_code == 200:
        body = r.json()
        assert "predicted_yield" in body
        assert "confidence_interval" in body
    else:
        assert r.status_code == 503


@pytest.mark.asyncio
async def test_optimize_smoke(client: AsyncClient):
    r = await client.post(
        "/api/optimize",
        json={
            "target_yield": 0.5,
            "tool_id": "tool_B",
            "constraints": {"temperature": {"max": 430.0}},
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "suggested_params" in data
    assert "achieved_yield" in data
