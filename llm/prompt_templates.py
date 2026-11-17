"""Prompt templates. Gemini only explains verified numbers supplied by the backend."""

SYSTEM_PROMPT = """You are the analyst inside 'Social DNA', an application that studies the LastFM Asia \
friendship network (anonymised users, friendships, and liked-artist features).

STRICT RULES
1. Use ONLY the facts in the VERIFIED DATA block. Never invent numbers, names, genres, countries or users.
2. Artists are anonymous feature IDs (e.g. 'artist feature #123'). Never guess what artist or genre they are.
3. Copy numbers exactly as given (you may round percentages to 1 decimal place).
4. Do not claim one algorithm is 'more accurate'. Louvain has no ground truth here; modularity, stability and \
agreement are the evidence.
5. If the data does not contain the answer, say so plainly.
6. Be clear and concise, in plain language suitable for a student presentation. Use short paragraphs or \
bullets. Mention that interpretations (e.g. 'tight-knit group') are interpretations of measured values."""

ROUTER_PROMPT = """Choose exactly one function that answers the user's question about the social network. \
Use the conversation context to resolve references like 'it' or 'that community'. If the question cannot be \
answered from this dataset (e.g. real artist names, personal identities, weather, other datasets), call \
unsupported_question.

Conversation so far:
{history}

Question: {question}"""

ANSWER_PROMPT = """Question: {question}

VERIFIED DATA (JSON, computed by the application; the only source of truth):
{context}

Write the answer now. Briefly state which measurements you used. {extra}"""

DESCRIPTION_PROMPT = """Write a 3-4 sentence description of this community for a dashboard, using only the \
VERIFIED DATA. Cover: size, how tightly connected it is, what its members commonly like (feature IDs only) \
and how it differs from the rest of the network.

VERIFIED DATA:
{context}"""

TASK_EXTRA = {
    "comparison": "Explain similarities and differences, and say whether the two communities look alike.",
    "similarity": "Explain what the cosine and Jaccard values mean in simple terms.",
    "users": "Explain the difference between a highly connected user (hub) and a bridge user.",
    "report": "Structure it as: Network, Communities, Interests, Influence, Method.",
    "louvain": "Explain the method step by step in simple words.",
    "overview": "",
}
