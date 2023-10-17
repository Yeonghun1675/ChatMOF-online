import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from chatmof.agents.agent import ChatMOF
from langchain.callbacks import StdOutCallbackHandler
import io
import sys
import contextlib
import re
import time  # For simulating progress in the progress bar

verbose = True
search_internet = False

openai_api_key = st.text_input('Enter OpenAI api key below 👇')

if openai_api_key:
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    callback_manager = [StdOutCallbackHandler()]

    chatmof = ChatMOF.from_llm(
        llm=llm, 
        verbose=verbose, 
    )


@contextlib.contextmanager
def capture_stdout():
    class Stream:
        def write(self, s):
            s = re.sub(r".+?m", '', s)
            if re.search(r"\B> ", s):
                return
            s = re.sub(
                r"\s(?P<name>Action Input|Action|Observation|Final Thought|Thought|Input|Final Answer|Answer):", 
                lambda t: "\n`{}`:".format(t.group('name')), 
                s,
            )
            s = re.sub(r"\[(?P<name>.+?)\]", lambda t: '**[{}]**'.format(t.group('name')), s)

            st.write(s)

    old_stdout = sys.stdout
    sys.stdout = Stream()  # Redirect stdout
    try:
        yield
    finally:
        sys.stdout = old_stdout  # Restore stdout

st.title('Welcome to the ChatMOF 🤖')

# Creating columns for input and button
col1, col2 = st.columns((8, 1))

with col1:
    input = st.text_input(
        'Enter question 👇', 
        value='What is the surface area of JUKPAI?'
    )  # Removed placeholder text

with col2:
    start_button = st.button('Start')

if start_button:
    if not openai_api_key:
        st.warning('You have to enter a OpenAI api key')
    elif input:
        st.subheader('Running...')
        with capture_stdout():
            text = chatmof.run(input)
        
        # Displaying the final answer
        st.subheader('Final Answer')
        st.text_area('', value=text, height=100, max_chars=None, key=None)
    else:
        st.warning('Please enter a question.')