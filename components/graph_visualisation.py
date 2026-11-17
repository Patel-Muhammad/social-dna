"""Interactive network views. The full graph (7,624 nodes) would freeze the browser, so we always
render a bounded subgraph chosen by the user's filters."""
from __future__ import annotations

import json
import math

import networkx as nx
import pandas as pd

from components.theme import community_color

PANEL_JS = """
<div id="sd-panel" style="position:absolute;top:10px;right:10px;width:250px;background:#fff;border:1px solid #E5E7EB;
 border-radius:12px;padding:10px 12px;font:13px Inter,system-ui,sans-serif;color:#111827;box-shadow:0 2px 8px rgba(0,0,0,.08);">
 <b>User details</b><div id="sd-body" style="margin-top:6px;color:#6B7280">Click a node to see its measured values.</div></div>
<script>
 var SD_INFO = __INFO__;
 network.on("click", function(p) {
   var body = document.getElementById("sd-body");
   if (p.nodes.length === 0) { body.innerHTML = "Click a node to see its measured values."; return; }
   var i = SD_INFO[p.nodes[0]];
   body.innerHTML = i ? i.replace(/\\n/g, "<br>") : "No details";
 });
</script>
"""


def select_subgraph(graph: nx.Graph, partition: dict[int, int], *, communities: list[int] | None = None,
                    focus_user: int | None = None, depth: int = 1, max_nodes: int = 400) -> nx.Graph:
    """Pick at most `max_nodes` nodes: an ego network around a user, or the highest-degree members of
    the selected communities (quota shared proportionally)."""
    if focus_user is not None:
        if focus_user not in graph:
            raise KeyError(focus_user)
        ego = nx.ego_graph(graph, focus_user, radius=depth)
        if ego.number_of_nodes() > max_nodes:
            keep = {focus_user} | set(sorted(ego.nodes, key=lambda n: -graph.degree(n))[: max_nodes - 1])
            ego = ego.subgraph(keep)
        return graph.subgraph(ego.nodes).copy()
    wanted = set(communities) if communities else set(partition.values())
    groups: dict[int, list[int]] = {}
    for node, c in partition.items():
        if c in wanted:
            groups.setdefault(c, []).append(node)
    total = sum(len(v) for v in groups.values())
    chosen: list[int] = []
    for c, members in groups.items():
        quota = len(members) if total <= max_nodes else max(5, int(max_nodes * len(members) / total))
        chosen += sorted(members, key=lambda n: -graph.degree(n))[:quota]
    return graph.subgraph(chosen).copy()


def render_network_html(sub: nx.Graph, partition: dict[int, int], users: pd.DataFrame,
                        highlight: int | None = None, focus_user: int | None = None,
                        height: int = 640, label_top: int = 40) -> str:
    from pyvis.network import Network
    net = Network(height=f"{height}px", width="100%", bgcolor="#FFFFFF", font_color="#111827",
                  cdn_resources="in_line", notebook=False)
    by_degree = sorted(sub.nodes, key=lambda n: -sub.degree(n))
    labelled = set(by_degree[:label_top]) | ({focus_user} if focus_user else set())
    info = {}
    for n in sub.nodes:
        c = partition[n]
        dim = highlight is not None and c != highlight
        row = users.loc[n] if n in users.index else None
        tip = f"User {n}\nCommunity {c}\nFriends in dataset: {sub.degree(n)} (shown) / " + \
              (f"{int(row.degree)} (total)" if row is not None else "?")
        if row is not None:
            tip += (f"\nPageRank: {row.pagerank:.5f}\nBetweenness (approx): {row.betweenness_centrality:.5f}"
                    f"\nCommunities reached: {int(row.neighbour_communities)}")
        info[str(n)] = tip
        net.add_node(int(n), label=str(n) if n in labelled else " ", title=tip,
                     color="#D1D5DB" if dim else community_color(c),
                     size=7 + 3.2 * math.sqrt(sub.degree(n)) + (14 if n == focus_user else 0),
                     borderWidth=4 if n == focus_user else 1,
                     font={"size": 14 if n in labelled else 0})
    for u, v in sub.edges():
        same = partition[u] == partition[v]
        net.add_edge(int(u), int(v), color="rgba(100,116,139,0.28)" if same else "rgba(217,119,6,0.55)", width=1)
    net.set_options(json.dumps({
        "physics": {"solver": "forceAtlas2Based", "forceAtlas2Based": {"gravitationalConstant": -45, "springLength": 90},
                    "stabilization": {"iterations": 180}},
        "interaction": {"hover": True, "tooltipDelay": 80, "navigationButtons": True, "zoomView": True}}))
    page = net.generate_html()
    return page.replace("</body>", PANEL_JS.replace("__INFO__", json.dumps(info)) + "</body>")
