import pandas as pd
import streamlit as st

from components import charts
from components.context import get_context
from components.metric_cards import feature_chips, metric_cards
from components.sidebar import render_sidebar
from components.theme import callout, page_header
from llm.gemini_client import GeminiUnavailableError

ctx = get_context()
render_sidebar(ctx)
svc = ctx.service
ids = svc.community_ids()
page_header("Community Comparison", "Put two communities side by side: size, shared interests, similarity and connectivity.")

c1, c2 = st.columns(2)
a = c1.selectbox("Community A", ids, index=1 if len(ids) > 1 else 0, format_func=lambda c: f"Community {c}")
b = c2.selectbox("Community B", ids, index=3 if len(ids) > 3 else 0, format_func=lambda c: f"Community {c}")
if a == b:
    st.info("Pick two different communities to compare.")
    st.stop()
r = svc.compare_communities(a, b)
A, B = r["community_a"], r["community_b"]
metric_cards([
    ("Cosine similarity", f"{r['cosine_similarity']:.3f}", "TF-IDF community profiles, 0–1"),
    ("Jaccard similarity", f"{r['jaccard_similarity']:.3f}", "overlap of top-30 common features"),
    ("Shared top features", f"{r['shared_common_feature_count']} / 30", "in both top-30 lists"),
    ("Friendships between them", f"{r['cross_edges']:,}", "cross-community links"),
])
st.write("")
g1, g2, g3 = st.columns(3)
names = (f"Community {a}", f"Community {b}")
g1.plotly_chart(charts.paired_bars(names, (A["member_count"], B["member_count"]), "Members"), use_container_width=True)
g2.plotly_chart(charts.paired_bars(names, (A["internal_edges"], B["internal_edges"]), "Internal friendships"), use_container_width=True)
g3.plotly_chart(charts.paired_bars(names, (A["avg_internal_degree"], B["avg_internal_degree"]), "Avg internal degree", ".1f"),
                use_container_width=True)
st.dataframe(pd.DataFrame({"Metric": ["Members", "% of network", "Internal friendships", "External friendships",
                                      "Internal density", "Conductance", "Within-community cosine", "Within-community Jaccard"],
                           names[0]: [f"{A['member_count']:,}", f"{A['percentage_of_network']:.1f}%", f"{A['internal_edges']:,}",
                                      f"{A['external_edges']:,}", f"{A['internal_density']:.4f}", f"{A['conductance']:.3f}",
                                      f"{A['within_community_cosine']:.3f}", f"{A['within_community_jaccard']:.3f}"],
                           names[1]: [f"{B['member_count']:,}", f"{B['percentage_of_network']:.1f}%", f"{B['internal_edges']:,}",
                                      f"{B['external_edges']:,}", f"{B['internal_density']:.4f}", f"{B['conductance']:.3f}",
                                      f"{B['within_community_cosine']:.3f}", f"{B['within_community_jaccard']:.3f}"]}),
             hide_index=True, use_container_width=True)

st.subheader("Features")
x, y, z = st.columns(3)
with x:
    st.markdown(f"**Only in Community {a}'s top 30**")
    feature_chips(r["only_a_common_features"])
with y:
    st.markdown("**Shared by both**")
    feature_chips([{"label": f["label"], "prevalence": (f["prevalence_a"] + f["prevalence_b"]) / 2, "lift": 0}
                   for f in r["shared_common_features"][:8]])
    st.caption("Chip % = average prevalence across both communities.")
with z:
    st.markdown(f"**Only in Community {b}'s top 30**")
    feature_chips(r["only_b_common_features"])

st.subheader("AI-generated explanation")
if st.button("Explain this comparison with Gemini", disabled=not ctx.analyst.gemini.is_configured):
    with st.spinner("Asking Gemini to explain verified results..."):
        try:
            st.markdown(ctx.analyst.explain(f"Compare community {a} and community {b}", r))
        except GeminiUnavailableError as exc:
            st.error(str(exc))
if not ctx.analyst.gemini.is_configured:
    callout("Gemini is not configured. Add <code>GEMINI_API_KEY</code> to <code>.env</code> to enable explanations.", "warn")
