import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from rag_backend import get_rag_chain, ask_question

st.set_page_config(page_title="Customer Support Assistant", layout="centered")
st.title("4MATION Customer Support Chat")

# 1. Initialize session state safely
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def load_chain():
    return get_rag_chain()


chain = load_chain()

# 2. Render past messages from history
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(message.content)

# 3. Handle user query
if user_query := st.chat_input("Ask a question..."):
    # Immediately render and persist the user question
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append(HumanMessage(content=user_query))

    # Stream the assistant response
    with st.chat_message("assistant"):
        # Pass a copy of the history *excluding* the latest user message
        # because the prompt template injects ("human", "{question}") separately
        history_for_chain = st.session_state.messages[:-1]

        response_stream = ask_question(chain, user_query, history_for_chain)
        full_response = st.write_stream(response_stream)

    # Persist the assistant response
    if full_response:
        st.session_state.messages.append(AIMessage(content=str(full_response)))