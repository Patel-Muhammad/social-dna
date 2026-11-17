"""Cached application context shared by all pages."""
from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from analysis.queries import AnalysisService
from core import config
from core.cache_manager import PipelineState, ensure_pipeline
from database.repositories import AnalysisRepository, ChatRepository
from llm.gemini_client import GeminiClient
from llm.query_router import Analyst


@dataclass
class AppContext:
    state: PipelineState
    service: AnalysisService
    repo: AnalysisRepository
    analyst: Analyst


@st.cache_resource(show_spinner=False)
def _service(db_path: str, run_id: int, _network) -> AnalysisService:
    return AnalysisService(AnalysisRepository(db_path), run_id, _network)


@st.cache_resource(show_spinner=False)
def _gemini() -> GeminiClient:
    return GeminiClient()


def get_context() -> AppContext:
    """Return the app context, running the first-time pipeline (with progress) if needed."""
    seed = st.session_state.get("seed", config.DEFAULT_SEED)
    resolution = st.session_state.get("resolution", config.DEFAULT_RESOLUTION)
    if "ctx_key" not in st.session_state or st.session_state.ctx_key != (seed, resolution):
        with st.status("Preparing Social DNA (first launch can take about a minute)...", expanded=True) as status:
            bar = st.progress(0.0)

            def progress(message: str, fraction: float) -> None:
                bar.progress(min(max(fraction, 0.0), 1.0))
                st.write(message)

            try:
                state = ensure_pipeline(seed=seed, resolution=resolution, progress=progress,
                                        force=st.session_state.pop("force_rerun", False))
            except Exception as exc:  # shown to the user with guidance, then stop the page
                status.update(label="Setup failed", state="error")
                st.error(f"{type(exc).__name__}: {exc}")
                st.stop()
            status.update(label="Ready", state="complete", expanded=False)
        st.session_state.ctx_key = (seed, resolution)
        st.session_state.pipeline_state = state
    state: PipelineState = st.session_state.pipeline_state
    service = _service(str(state.db_path), state.run_id, state.network)
    repo = AnalysisRepository(state.db_path)
    return AppContext(state, service, repo, Analyst(service, _gemini()))


def chat_repo() -> ChatRepository:
    return ChatRepository(config.DB_PATH)
