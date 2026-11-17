import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from components import charts
from components.context import get_context
from components.graph_visualisation import render_network_html, select_subgraph
from components.sidebar import render_sidebar
from components.theme import callout, community_color, page_header

ctx = get_context()
render_sidebar(ctx)
svc = ctx.service
graph = ctx.state.network.graph
partition = svc.partition
ids = svc.community_ids()

page_header("Community Explorer", "Explore the friendship graph. Nodes are users coloured by community; "
            "orange links cross community borders.")
tab_graph, tab_map = st.tabs(["User network", "Community map"])

with tab_graph:
    f1, f2, f3 = st.columns([3, 2, 2])
    chosen = f1.multiselect("Filter communities", ids, default=ids[:3], format_func=lambda c: f"Community {c}",
                            help="Only members of these communities are drawn. Leave empty to sample all communities.")
    highlight = f2.selectbox("Highlight community", [None] + ids, format_func=lambda c: "None" if c is None else f"Community {c}")
    max_nodes = f3.slider("Max nodes drawn", 50, 800, 300, 50, help="Keeps the browser responsive.")
    s1, s2 = st.columns([3, 2])
    query = s1.text_input("Search a user ID", placeholder="e.g. 7199", help="Shows the user and their friends.")
    depth = s2.radio("Friends depth", [1, 2], horizontal=True, help="1 = direct friends, 2 = friends of friends (capped).")

    focus = None
    if query.strip():
        if query.strip().isdigit() and int(query) in graph:
            focus = int(query)
        else:
            st.warning(f"User '{query}' was not found. Dataset IDs are integers between "
                       f"{int(ctx.state.network.node_ids.min())} and {int(ctx.state.network.node_ids.max())}.")
    sub = select_subgraph(graph, partition, communities=chosen or None, focus_user=focus, depth=depth, max_nodes=max_nodes)
    if sub.number_of_nodes() == 0:
        st.info("Nothing to draw. Pick at least one community.")
    else:
        if focus is not None:
            d = svc.get_user_details(focus)
            callout(f"<b>User {focus}</b> · community {d['community']} · {d['degree']} friends "
                    f"({d['external_degree']} outside their community, across {d['neighbour_communities']} communities) "
                    f"· PageRank {d['pagerank']:.5f}")
        st.caption(f"Drawing {sub.number_of_nodes():,} of {graph.number_of_nodes():,} users and {sub.number_of_edges():,} "
                   "friendships. Scroll to zoom, drag to pan, hover for details, click a node for its panel.")
        with st.spinner("Laying out the network..."):
            html = render_network_html(sub, partition, svc.users, highlight=highlight, focus_user=focus)
        components.html(html, height=670, scrolling=False)
        present = sorted({partition[n] for n in sub.nodes})
        st.markdown(" ".join(f'<span class="feature-chip"><span style="color:{community_color(c)}">●</span> Community {c}</span>'
                             for c in present), unsafe_allow_html=True)

with tab_map:
    min_edges = st.slider("Minimum friendships between communities to draw a link", 1, 100, 10)
    st.plotly_chart(charts.community_map(svc.communities, svc.similarity, min_edges), use_container_width=True)
    st.caption("Each circle is a community (size = members). Links show how many friendships cross between two communities.")
    st.plotly_chart(charts.similarity_heatmap(svc.similarity, ids), use_container_width=True)
