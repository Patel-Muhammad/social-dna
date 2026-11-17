"""Social DNA - entry point and navigation. Run with:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="Social DNA", page_icon=":material/hub:", layout="wide",
                   initial_sidebar_state="expanded")

from components.theme import inject_css  # noqa: E402

inject_css()
navigation = st.navigation([
    st.Page("pages/1_Overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("pages/2_Community_Explorer.py", title="Community Explorer", icon=":material/hub:"),
    st.Page("pages/3_Community_Profiles.py", title="Community Profiles", icon=":material/groups:"),
    st.Page("pages/4_Community_Comparison.py", title="Community Comparison", icon=":material/compare_arrows:"),
    st.Page("pages/5_Influence_Analysis.py", title="Influence Analysis", icon=":material/trending_up:"),
    st.Page("pages/6_AI_Analyst.py", title="AI Community Analyst", icon=":material/smart_toy:"),
    st.Page("pages/7_Methodology.py", title="Methodology & Dataset", icon=":material/menu_book:"),
])
navigation.run()
