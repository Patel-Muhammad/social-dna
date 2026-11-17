from __future__ import annotations

import streamlit as st

from core import config


def render_sidebar(ctx) -> None:
    with st.sidebar:
        st.markdown('<div class="brand"><div class="logo">S</div><div><div class="name">Social DNA</div>'
                    '<div class="tag">AI-based community discovery</div></div></div>', unsafe_allow_html=True)
        st.divider()
        run = ctx.service.run
        st.markdown(f"**Active analysis**  \nLouvain · seed {run['random_seed']} · resolution {run['resolution']:g}  \n"
                    f"{run['community_count']} communities · modularity {run['modularity']:.3f}")
        gem = ctx.analyst.gemini
        state = "not configured" if not gem.is_configured else ("unreachable (retrying later)" if gem.is_down else "ready")
        st.markdown(f'<span class="pill {"on" if state == "ready" else "off"}">Gemini {state}</span>',
                    unsafe_allow_html=True)
        if not gem.is_configured:
            st.caption("Add GEMINI_API_KEY to your .env file to enable AI explanations. The rest works without it.")
        with st.expander("Re-run community detection"):
            seed = st.number_input("Random seed", 0, 10_000, int(st.session_state.get("seed", config.DEFAULT_SEED)))
            res = st.slider("Resolution", 0.5, 2.0, float(st.session_state.get("resolution", config.DEFAULT_RESOLUTION)), 0.1,
                            help="Higher values give more, smaller communities.")
            if st.button("Run analysis", use_container_width=True):
                st.session_state.seed, st.session_state.resolution = int(seed), float(res)
                st.rerun()
        st.divider()
        st.caption("Data: LastFM Asia Social Network (SNAP). Users are anonymised IDs.")
