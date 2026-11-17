from __future__ import annotations

import html

import streamlit as st


def metric_cards(items: list[tuple[str, str, str | None]], columns: int | None = None) -> None:
    """items = [(label, value, hint)] rendered as a responsive row of cards."""
    cols = st.columns(columns or len(items))
    for col, (label, value, hint) in zip(cols, items):
        hint_html = f'<div class="hint">{html.escape(hint)}</div>' if hint else ""
        col.markdown(
            f'<div class="card"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{html.escape(str(value))}</div>{hint_html}</div>',
            unsafe_allow_html=True)


def feature_chips(features: list[dict], show: str = "prevalence") -> None:
    if not features:
        st.caption("No features meet the minimum-support rule for this community.")
        return
    chips = []
    for f in features:
        extra = f"{f['prevalence']:.0%}" if show == "prevalence" else f"lift {f['lift']:.1f}x"
        chips.append(f'<span class="feature-chip">{html.escape(f["label"])} · {extra}</span>')
    st.markdown("".join(chips), unsafe_allow_html=True)
