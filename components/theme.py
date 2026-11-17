"""Visual identity: CSS, palette and a deterministic colour per community."""
from __future__ import annotations

import colorsys

import streamlit as st

INDIGO, TEAL, AMBER, ROSE, SLATE = "#4F46E5", "#0D9488", "#D97706", "#E11D48", "#64748B"
INK, MUTED, BORDER = "#111827", "#6B7280", "#E5E7EB"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }
h1 { font-weight: 700; letter-spacing: -0.02em; font-size: 2rem !important; }
h2, h3 { font-weight: 600; letter-spacing: -0.01em; }
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E5E7EB; }
.brand { display:flex; align-items:center; gap:.6rem; margin: .2rem 0 1rem 0; }
.brand .logo { width:34px; height:34px; border-radius:10px; background: linear-gradient(135deg,#4F46E5,#0D9488);
  display:flex; align-items:center; justify-content:center; color:#fff; font-weight:700; }
.brand .name { font-weight:700; font-size:1.1rem; line-height:1.1; } .brand .tag { color:#6B7280; font-size:.75rem; }
.page-sub { color:#6B7280; margin-top:-.6rem; margin-bottom:1.4rem; font-size:1.02rem; }
.card { background:#fff; border:1px solid #E5E7EB; border-radius:14px; padding:1rem 1.15rem; height:100%;
  box-shadow: 0 1px 2px rgba(16,24,40,.04); }
.card .label { color:#6B7280; font-size:.78rem; font-weight:500; text-transform:uppercase; letter-spacing:.04em; }
.card .value { font-size:1.75rem; font-weight:700; letter-spacing:-0.02em; margin-top:.15rem; color:#111827; }
.card .hint { color:#6B7280; font-size:.78rem; margin-top:.1rem; }
.callout { background:#EEF2FF; border:1px solid #C7D2FE; border-radius:12px; padding:.85rem 1.1rem; color:#312E81; margin:.4rem 0 1rem 0; }
.callout.warn { background:#FFFBEB; border-color:#FDE68A; color:#78350F; }
.callout.good { background:#F0FDFA; border-color:#99F6E4; color:#134E4A; }
.pill { display:inline-block; padding:.12rem .6rem; border-radius:999px; font-size:.75rem; font-weight:600; margin-right:.3rem; }
.pill.on { background:#DCFCE7; color:#166534; } .pill.off { background:#FEE2E2; color:#991B1B; }
.feature-chip { display:inline-block; background:#F3F4F6; border:1px solid #E5E7EB; border-radius:8px; padding:.15rem .55rem;
  margin:.15rem .25rem .15rem 0; font-size:.82rem; }
.small { color:#6B7280; font-size:.82rem; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #E5E7EB; border-radius:12px; padding:.7rem 1rem; }
.stTabs [data-baseweb="tab"] { font-weight:600; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def callout(text: str, kind: str = "info") -> None:
    cls = {"info": "", "warn": " warn", "good": " good"}[kind]
    st.markdown(f'<div class="callout{cls}">{text}</div>', unsafe_allow_html=True)


def community_color(community: int) -> str:
    """Golden-ratio hue walk: visually distinct, stable colour for any community number."""
    hue = (community * 0.61803398875) % 1.0
    sat = 0.62 + 0.12 * ((community * 7) % 3) / 2
    light = 0.46 + 0.08 * ((community * 5) % 3) / 2
    r, g, b = colorsys.hls_to_rgb(hue, light, sat)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
