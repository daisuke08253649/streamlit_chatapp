import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks import get_openai_callback, StreamlitCallbackHandler

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def init_page():
    st.set_page_config(page_title="ChatApp", page_icon="☺")
    st.header("ChatApp")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        st.session_state.costs = []


def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4", "GPT-4o"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    elif model == "GPT-4":
        model_name = "gpt-4"
    else:
        model_name = "gpt-4o"

    temperature = st.sidebar.slider(
        "Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.01
    )

    return ChatOpenAI(temperature=temperature, model_name=model_name, streaming=True)


def get_answer(llm, messages):
    with get_openai_callback() as cb:
        answer = llm(messages)

    return answer.content, cb.total_cost


def main():
    init_page()

    llm = select_model()
    init_messages()

    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

        else:
            st.write(f"System message: {message.content}")

    user_input = st.chat_input("プロンプトを入力してください")
    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        st.chat_message("user").markdown(user_input)
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            response = llm(messages, callbacks=[st_callback])
        st.session_state.messages.append(AIMessage(content=response.content))


if __name__ == "__main__":
    main()
