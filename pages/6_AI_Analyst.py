import streamlit as st

from components.chat_interface import render_result
from components.context import chat_repo, get_context
from components.sidebar import render_sidebar
from components.theme import callout, page_header

ctx = get_context()
render_sidebar(ctx)
repo = chat_repo()
page_header("AI Community Analyst", "Ask questions in plain English. Each question is routed to a verified backend "
            "function; the answer is built from stored calculations, never invented.")

if "chat_session" not in st.session_state:
    st.session_state.chat_session = repo.create_session()
sid = st.session_state.chat_session

with st.sidebar:
    st.divider()
    st.markdown("**Conversations**")
    if st.button("New conversation", use_container_width=True):
        st.session_state.chat_session = repo.create_session()
        st.rerun()
    past = repo.sessions().head(8)
    for r in past.itertuples():
        if r.id != sid and st.button(f"{r.title[:28]}", key=f"s{r.id}", use_container_width=True):
            st.session_state.chat_session = r.id
            st.rerun()

if not ctx.analyst.gemini.is_configured:
    callout("Gemini is not configured, so answers are plain summaries of verified numbers. Add <code>GEMINI_API_KEY</code> "
            "to <code>.env</code> and restart for natural-language explanations.", "warn")

history = repo.messages(sid)
SUGGESTIONS = ["Discover all the communities.", "Which is the largest community?", "Explain community 5.",
               "What interests are common in community 3?", "Compare community 2 and community 4.",
               "Which communities are most similar?", "Find users who connect different communities.",
               "Give me a report on the entire network.", "Explain how Louvain discovered these communities."]
if not history:
    st.markdown("**Try a question**")
    cols = st.columns(3)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 3].button(s, key=f"sg{i}", use_container_width=True):
            st.session_state.pending_q = s
            st.rerun()

for i, m in enumerate(history):
    with st.chat_message(m["role"]):
        st.markdown(m["message"])
        meta = m.get("meta") or {}
        if m["role"] == "assistant":
            if meta.get("notice"):
                st.caption(f"⚠ {meta['notice']}")
            render_result(meta.get("result"), ctx, key=f"h{sid}-{i}")

question = st.chat_input("Ask about communities, interests, influential users...") or st.session_state.pop("pending_q", None)
if question:
    ctxt = [{"role": m["role"], "text": m["message"], "params": (m.get("meta") or {}).get("params")} for m in history]
    repo.add_message(sid, "user", question)
    if not history:
        repo.rename_session(sid, question)
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Calling verified analysis functions..."):
            resp = ctx.analyst.answer(question, ctxt)
        st.markdown(resp.text)
        if resp.notice:
            st.caption(f"⚠ {resp.notice}")
        render_result(resp.result, ctx, key=f"new{sid}-{len(history)}")
        if resp.function:
            st.caption(f"Function: `{resp.function}({', '.join(f'{k}={v}' for k, v in resp.params.items())})` · "
                       f"{'explained by Gemini' if resp.used_llm else 'plain summary'}")
    meta = {"function": resp.function, "params": resp.params, "notice": resp.notice, "used_llm": resp.used_llm}
    if resp.result is not None and len(str(resp.result)) < 60_000:
        meta["result"] = resp.result
    repo.add_message(sid, "assistant", resp.text, meta)
