import pandas as pd
import plotly.express as px
import streamlit as st

from components import charts
from components.context import get_context
from components.sidebar import render_sidebar
from components.theme import callout, page_header

ctx = get_context()
render_sidebar(ctx)
svc = ctx.service
page_header("Influence Analysis", "Who is most connected, most central, and who links different communities. "
            "User IDs are anonymised dataset identifiers, not real identities.")
callout("<b>Hub vs bridge.</b> A <b>hub</b> has many friends, often all in the same community. A <b>bridge</b> has friends in "
        "<i>several</i> communities and lies on shortest paths between them, so removing them could separate groups. "
        "A user can be both, but usually they are different people.")
n = st.slider("Users per ranking", 5, 30, 10)
res = svc.get_influential_users(n)
note = res.get("betweenness_note")
tabs = st.tabs(["Degree", "Betweenness", "Closeness", "PageRank", "Bridge users"])
spec = [("degree", "Friends (degree)", "Most connected users: number of direct friendships."),
        ("betweenness_centrality", "Betweenness centrality", "Share of shortest paths passing through the user. "
         f"{note or ''}"),
        ("closeness_centrality", "Closeness centrality", "How close the user is to everyone else (exact)."),
        ("pagerank", "PageRank", "Importance by being friends with other important users (exact).")]
for tab, (col, label, text) in zip(tabs[:4], spec):
    with tab:
        st.caption(text)
        df = pd.DataFrame(res[f"top_by_{col}"])
        st.plotly_chart(charts.ranking_bar(df, col, label), use_container_width=True, key=f"rank-{col}")
        st.dataframe(df, hide_index=True, use_container_width=True)
with tabs[4]:
    br = svc.get_bridge_users(n)
    st.caption(br["definition"])
    st.dataframe(pd.DataFrame(br["users"]).drop(columns=["explanation"]), hide_index=True, use_container_width=True)
    st.markdown("**Why each user matters**")
    for u in br["users"]:
        st.markdown(f"- {u['explanation']}")
    allu = svc.users.reset_index()
    fig = px.scatter(allu, x="degree", y="betweenness_centrality", color="participation_coefficient",
                     color_continuous_scale="Viridis", opacity=0.6, hover_data=["user_id", "community"],
                     labels={"degree": "Friends (degree)", "betweenness_centrality": "Betweenness (approx.)",
                             "participation_coefficient": "Participation"})
    fig.update_xaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Top-left = bridges (high betweenness for their degree). Right = hubs. Colour = how evenly a user's friends "
               "are spread over communities.")
