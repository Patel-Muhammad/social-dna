"""Plotly figures with one consistent look."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from components.theme import BORDER, INDIGO, INK, MUTED, TEAL, community_color

LAYOUT = dict(font=dict(family="Inter, system-ui, sans-serif", color=INK, size=13),
              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              margin=dict(l=10, r=10, t=40, b=10), hoverlabel=dict(font_size=13))


def _style(fig: go.Figure, title: str | None = None, height: int = 360) -> go.Figure:
    fig.update_layout(**LAYOUT, title=dict(text=title, font=dict(size=15)) if title else None, height=height)
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


def community_size_bar(communities: pd.DataFrame, highlight: int | None = None) -> go.Figure:
    df = communities.sort_values("community_number")
    colors = [community_color(int(c)) for c in df.community_number]
    fig = go.Figure(go.Bar(
        x=df.community_number.astype(str), y=df.member_count, marker_color=colors,
        marker_line_width=[2 if highlight == c else 0 for c in df.community_number], marker_line_color=INK,
        hovertemplate="Community %{x}<br>%{y:,} members<extra></extra>"))
    fig.update_xaxes(title="Community", type="category")
    fig.update_yaxes(title="Members")
    return _style(fig, "Community sizes")


def size_distribution(communities: pd.DataFrame) -> go.Figure:
    fig = px.histogram(communities, x="member_count", nbins=15, color_discrete_sequence=[INDIGO])
    fig.update_xaxes(title="Members per community")
    fig.update_yaxes(title="Number of communities")
    return _style(fig, "Community size distribution")


def community_share_donut(communities: pd.DataFrame, top: int = 10) -> go.Figure:
    df = communities.sort_values("member_count", ascending=False)
    head, tail = df.head(top), df.iloc[top:]
    labels = [f"Community {c}" for c in head.community_number] + (["Other communities"] if len(tail) else [])
    values = list(head.member_count) + ([int(tail.member_count.sum())] if len(tail) else [])
    colors = [community_color(int(c)) for c in head.community_number] + ([MUTED] if len(tail) else [])
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.55, marker=dict(colors=colors),
                           sort=False, textinfo="percent", hovertemplate="%{label}<br>%{value:,} users<extra></extra>"))
    return _style(fig, f"Share of users (top {top} communities)")


def degree_distribution(degrees: pd.Series) -> go.Figure:
    counts = degrees.value_counts().sort_index()
    fig = go.Figure(go.Scatter(x=counts.index, y=counts.values, mode="markers", marker=dict(color=TEAL, size=7),
                               hovertemplate="degree %{x}: %{y} users<extra></extra>"))
    fig.update_xaxes(title="Degree (friends)", type="log")
    fig.update_yaxes(title="Users", type="log")
    return _style(fig, "Degree distribution (log-log)")


def community_map(communities: pd.DataFrame, similarity: pd.DataFrame, min_edges: int = 5) -> go.Figure:
    """Communities as nodes (size = members), edges = friendships between communities."""
    import math
    ids = list(communities.community_number)
    pos = {c: (math.cos(2 * math.pi * i / len(ids)), math.sin(2 * math.pi * i / len(ids))) for i, c in enumerate(ids)}
    fig = go.Figure()
    for r in similarity[similarity.cross_edges >= min_edges].itertuples():
        (x0, y0), (x1, y1) = pos[r.community_a], pos[r.community_b]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="text",
                                 text=f"{r.community_a} - {r.community_b}: {r.cross_edges} friendships",
                                 line=dict(width=0.4 + r.cross_edges ** 0.5 * 0.45, color="rgba(100,116,139,0.35)"),
                                 showlegend=False))
    fig.add_trace(go.Scatter(
        x=[pos[c][0] for c in ids], y=[pos[c][1] for c in ids], mode="markers+text",
        text=[str(c) for c in ids], textfont=dict(color="white", size=11),
        marker=dict(size=[14 + m ** 0.5 * 1.1 for m in communities.member_count],
                    color=[community_color(int(c)) for c in ids], line=dict(width=1.5, color="white")),
        hovertext=[f"Community {c}: {m:,} members" for c, m in zip(ids, communities.member_count)],
        hoverinfo="text", showlegend=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    return _style(fig, f"Community map (links shown when ≥ {min_edges} friendships)", 520)


def similarity_heatmap(sim: pd.DataFrame, ids: list[int]) -> go.Figure:
    import numpy as np
    mat = np.eye(len(ids))
    pos = {c: i for i, c in enumerate(ids)}
    for r in sim.itertuples():
        mat[pos[r.community_a], pos[r.community_b]] = mat[pos[r.community_b], pos[r.community_a]] = r.cosine_similarity
    fig = go.Figure(go.Heatmap(z=mat, x=[str(i) for i in ids], y=[str(i) for i in ids], colorscale="Purples",
                               hovertemplate="Community %{y} vs %{x}<br>cosine %{z:.3f}<extra></extra>",
                               colorbar=dict(title="cosine")))
    fig.update_xaxes(type="category", title="Community")
    fig.update_yaxes(type="category", title="Community", autorange="reversed")
    return _style(fig, "Interest similarity between communities (cosine, TF-IDF)", 560)


def feature_bar(features: list[dict], title: str, value: str = "prevalence", color: str = INDIGO,
                compare: list[dict] | None = None) -> go.Figure:
    df = pd.DataFrame(features)
    if df.empty:
        return _style(go.Figure(), title, 120)
    fig = go.Figure(go.Bar(y=df.label, x=df[value], orientation="h", marker_color=color,
                           customdata=df[["frequency", "network_prevalence", "lift"]].to_numpy(),
                           hovertemplate="%{y}<br>prevalence %{x:.1%}<br>%{customdata[0]} members<br>"
                                         "network prevalence %{customdata[1]:.1%}<br>lift %{customdata[2]:.2f}x<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickformat=".0%", title="Share of community members")
    return _style(fig, title, 120 + 26 * len(df))


def ranking_bar(df: pd.DataFrame, value: str, label: str, color: str = INDIGO) -> go.Figure:
    d = df.copy()
    d["user"] = "User " + d.user_id.astype(str) + " (c" + d.community.astype(str) + ")"
    fig = go.Figure(go.Bar(y=d.user, x=d[value], orientation="h", marker_color=color,
                           hovertemplate="%{y}<br>" + label + ": %{x:.4f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title=label)
    return _style(fig, label, 120 + 28 * len(d))


def paired_bars(names: tuple[str, str], values: tuple[float, float], title: str, fmt: str = ",") -> go.Figure:
    fig = go.Figure(go.Bar(x=list(names), y=list(values), marker_color=[INDIGO, TEAL], text=[f"{v:{fmt}}" for v in values],
                           textposition="outside"))
    fig.update_yaxes(rangemode="tozero")
    return _style(fig, title, 300)
