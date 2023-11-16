import streamlit as st
from openai.error import AuthenticationError
from langchain.chat_models import ChatOpenAI

from chatmof import ChatMOF
from utils import from_llm_revised, capture_stdout
from revised_tools.error import ChatMOFOnlineError


# st.set_page_config(layout="wide")

verbose = True
search_internet = False
default_openai_key = None

title = "🤖 Welcome to the ChatMOF"
description = """- **ChatMOF** is an autonomous Artificial Intelligence (AI) system that is built to predict and generate of metal-organic frameworks (MOFs).
- By leveraging a large-scale language model (**gpt-4**), ChatMOF extracts key details from textual inputs and delivers appropriate responses, thus eliminating the necessity for rigid structured queries.
- In online demo, **only the `Search task` is available**.  For `prediction tasks` and `generation tasks` that require machine learning do not work properly, except for the provided examples. If you want to test your own example, please use the code available on our [github](https://github.com/Yeonghun1675/ChatMOF).
- You need to enter the material name into **CoREMOF's REFCODE** (e.g. JUKPAI, XEGKUR, PITPEP).

## Start!
"""


def run_demo():
    ChatMOF.from_llm = from_llm_revised  # revise functions in ChatMOF

    with st.sidebar:
        st.header('OpenAI ChatModel')
        selected_model = st.selectbox(
            label="Choose your llm model",
            options=("gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k")
        )
        selected_temp = st.slider('Temperature', 0.0, 1.0, 0.1)

    openai_api_key = st.text_input(
        'Enter OpenAI api key below 👇', value=default_openai_key)

    if openai_api_key:
        llm = ChatOpenAI(
            temperature=selected_temp,
            model_name=selected_model,
            openai_api_key=openai_api_key
        )

        chatmof = ChatMOF.from_llm(
            llm=llm,
            verbose=verbose,
        )

    with open('questions.txt') as f:
        questions = [
            line.strip() for line in f
        ]

    st.title(title)
    st.write(description)

    col1, col2 = st.columns((8, 1))

    with col1:
        selected_question_index = st.selectbox(
            'Choose a question or enter your own 👇',
            options=[*questions, 'Custom question 🚀']
        )
    with col2:
        st.write("")
        start_button = st.button('Start')

    if selected_question_index == 'Custom question 🚀':
        input_question = st.text_input('Enter your question 👇')
    else:
        input_question = selected_question_index

    if start_button:
        if not openai_api_key:
            st.warning('You have to enter your OpenAI api key!')

        elif input_question:
            st.subheader('Running...')
            try:
                with capture_stdout():
                    text = chatmof.run(input_question)
                st.subheader('Final Answer')
                st.text_area('', value=text, height=100,
                             max_chars=None, key=None)
            except AuthenticationError:
                st.warning(
                    'Incorrect API key provied. You can find your API key at https://platform.openai.com/account/api-keys')

            except ChatMOFOnlineError as e:
                st.warning(str(e))

        else:
            st.warning('Please enter a question.')


if __name__ == '__main__':
    run_demo()
