"""Chat UI helpers: render an analyst response with its supporting statistics and charts."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components import charts
from components.metric_cards import feature_chips


def render_result(result: dict | None, ctx, key: str) -> None:
    """Show supporting statistics/charts for the backend result behind an answer."""
    if not result:
        return
    k = result.get("kind")
    with st.expander("Supporting statistics and charts", expanded=False):
        if k == "communities_table":
            df = pd.DataFrame(result["communities"])
            st.plotly_chart(charts.community_size_bar(df.rename(columns={"community_id": "community_number"})),
                            use_container_width=True, key=f"{key}-c")
            st.dataframe(df, use_container_width=True, hide_index=True)
        elif k in ("community", "interests"):
            st.plotly_chart(charts.feature_bar(result["common_features"][:10], "Most common features"),
                            use_container_width=True, key=f"{key}-a")
            st.plotly_chart(charts.feature_bar(result["distinctive_features"][:10], "Most distinctive features",
                                               color="#0D9488"), use_container_width=True, key=f"{key}-b")
        elif k == "comparison":
            a, b = result["community_a"], result["community_b"]
            st.plotly_chart(charts.paired_bars((f"Community {a['community_id']}", f"Community {b['community_id']}"),
                            (a["member_count"], b["member_count"]), "Members"), use_container_width=True, key=f"{key}-m")
            st.write({"cosine": round(result["cosine_similarity"], 4), "jaccard": round(result["jaccard_similarity"], 4),
                      "friendships between them": result["cross_edges"]})
        elif k == "similarity":
            st.write({"cosine": round(result["cosine_similarity"], 4), "jaccard": round(result["jaccard_similarity"], 4),
                      "friendships between them": result["cross_edges"]})
        elif k == "similar_pairs":
            st.dataframe(pd.DataFrame(result["pairs"]), use_container_width=True, hide_index=True)
        elif k == "users" and result["role"] == "bridge":
            st.dataframe(pd.DataFrame(result["users"]).drop(columns=["explanation"]), use_container_width=True, hide_index=True)
        elif k == "users":
            df = pd.DataFrame(result["top_by_pagerank"])
            st.plotly_chart(charts.ranking_bar(df, "pagerank", "PageRank"), use_container_width=True, key=f"{key}-p")
        elif k == "overview":
            st.dataframe(pd.DataFrame([{kk: vv for kk, vv in result.items() if not isinstance(vv, dict) and kk != "kind"}]).T
                         .rename(columns={0: "value"}), use_container_width=True)
        else:
            st.json(result, expanded=False)
        st.caption(f"Function called: `{result.get('kind')}` – values come from stored calculations, not from the AI.")
