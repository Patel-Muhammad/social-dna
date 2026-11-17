import pandas as pd
import streamlit as st

from components import charts
from components.context import get_context
from components.metric_cards import metric_cards
from components.sidebar import render_sidebar
from components.theme import callout, page_header

ctx = get_context()
render_sidebar(ctx)
svc, state = ctx.service, ctx.state
o = svc.get_network_overview()
comms = svc.communities

page_header("Overview", "What the LastFM Asia friendship network looks like and how Louvain split it into communities.")
metric_cards([
    ("Users", f"{o['users']:,}", "nodes in the graph"),
    ("Friendships", f"{o['friendships']:,}", "undirected edges"),
    ("Communities", f"{o['communities']}", "found by Louvain"),
    ("Largest community", f"{o['largest_community']['members']:,}", f"community {o['largest_community']['community_id']}"),
])
st.write("")
metric_cards([
    ("Average community size", f"{o['average_community_size']:.0f}", "members"),
    ("Modularity", f"{o['modularity']:.3f}", "0 = random, higher = clearer groups"),
    ("Avg clustering", f"{o['average_clustering']:.3f}", "friends of friends are friends"),
    ("Average degree", f"{o['average_degree']:.2f}", "friends per user"),
])
st.write("")
left, right = st.columns([3, 2])
left.plotly_chart(charts.community_size_bar(comms), use_container_width=True)
right.plotly_chart(charts.community_share_donut(comms), use_container_width=True)
left, right = st.columns(2)
left.plotly_chart(charts.size_distribution(comms), use_container_width=True)
right.plotly_chart(charts.degree_distribution(svc.users["degree"]), use_container_width=True)

st.subheader("Key analytical insights")
m = svc.metrics
sim = svc.similarity.sort_values("cosine_similarity", ascending=False).iloc[0]
top3 = comms.member_count.sort_values(ascending=False).head(3).sum() / o["users"]
insights = [
    f"**{o['communities']} communities** were found with modularity **{o['modularity']:.3f}**. Only "
    f"**{o['inter_community_edge_share']:.1%}** of friendships cross community borders.",
    f"The three largest communities hold **{top3:.1%}** of all users; the smallest has "
    f"**{o['smallest_community']['members']}** members.",
    f"The most similar pair by interests is communities **{int(sim.community_a)} and {int(sim.community_b)}** "
    f"(cosine similarity {sim.cosine_similarity:.3f}).",
    f"Across 10 random seeds Louvain's partitions agree with an average NMI of **{m['stability_pairwise_nmi_mean']:.3f}** "
    f"(modularity {m['stability_modularity_min']:.3f}–{m['stability_modularity_max']:.3f}).",
    f"Greedy Modularity found **{int(m['greedy_community_count'])}** communities (modularity {m['greedy_modularity']:.3f}); "
    f"its agreement with Louvain is NMI {m['louvain_vs_greedy_nmi']:.3f}. Higher modularity here does not prove a more "
    "accurate result, since no ground-truth communities exist.",
]
for line in insights:
    st.markdown(f"- {line}")

st.subheader("Dataset information")
v = state.validation
c1, c2 = st.columns(2)
c1.dataframe(pd.DataFrame({"Check": ["Users", "Friendship edges (clean)", "Duplicate edges removed", "Self-loops removed",
                                      "Isolated users", "Connected components", "Largest component",
                                      "Users with no artist features", "Distinct artist features",
                                      "Avg features per user"],
                           "Value": [f"{v['total_users']:,}", f"{v['total_edges_clean']:,}", v["duplicate_edges"],
                                     v["self_loops"], v["isolated_nodes"], v["connected_components"],
                                     f"{v['largest_component_size']:,}", v["users_without_features"],
                                     f"{v['feature_count']:,}", f"{v['mean_features_per_user']:.1f}"]}),
             hide_index=True, use_container_width=True)
with c2:
    callout("Source: <b>LastFM Asia Social Network</b> (SNAP / FEATHER, collected March 2020). Nodes are LastFM users from "
            "Asian countries; edges are mutual follower relationships; features are artists the user likes. "
            "User IDs are anonymised. See <i>Methodology &amp; Dataset</i> for licence and citation.")
    callout("Counts match the documented SNAP values." if v["matches_documented_counts"] else
            "Counts differ from the documented SNAP values; check the dataset files.",
            "good" if v["matches_documented_counts"] else "warn")
