"""FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from api.state import AppState


def get_app_state(request: Request) -> AppState:
    return request.app.state.app_state


AppStateDep = Annotated[AppState, Depends(get_app_state)]
