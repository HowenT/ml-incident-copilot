"""Thin HTTP client for the FastAPI backend."""
from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")


class ApiError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


@st.cache_resource
def _client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=120)


def _call(method: str, path: str, **kw) -> Any:
    try:
        r = _client().request(method, path, **kw)
    except httpx.HTTPError as e:
        raise ApiError(0, f"Cannot reach the API at {API_URL} ({e.__class__.__name__}). Is the backend running?") from e
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise ApiError(r.status_code, str(detail))
    return r.json()


def get(path: str, **params) -> Any:
    return _call("GET", path, params={k: v for k, v in params.items() if v is not None})


def post(path: str, body: dict | None = None, **params) -> Any:
    return _call("POST", path, json=body or {}, params=params or None)


def patch(path: str, body: dict) -> Any:
    return _call("PATCH", path, json=body)


# Monitoring reads change only on reseed; cache briefly to keep reruns snappy.
@st.cache_data(ttl=60, show_spinner=False)
def cached(path: str, **params) -> Any:
    return get(path, **params)


def clear_cache() -> None:
    cached.clear()


def guard(fn, *args, **kwargs):
    """Call the API and show a friendly error instead of a traceback."""
    try:
        return fn(*args, **kwargs)
    except ApiError as e:
        st.error(e.detail)
        return None
