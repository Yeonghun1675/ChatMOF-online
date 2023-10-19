import sys
import contextlib
import re

import streamlit as st

from langchain.agents import initialize_agent, AgentType
from chatmof.agents.prompt import PREFIX, FORMAT_INSTRUCTIONS, SUFFIX
from revised_tools.tool_utils import load_chatmof_tools_revised


@classmethod
def from_llm_revised(
    cls,
    llm,
    verbose: bool = False,
    search_internet: bool = True,
):
    tools = load_chatmof_tools_revised(
        llm=llm, 
        verbose=verbose, 
        search_internet=search_internet
    )
    
    agent_kwargs = {
        'prefix': PREFIX,
        'format_instructions': FORMAT_INSTRUCTIONS,
        'suffix': SUFFIX
    }
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=verbose,
        agent_kwargs=agent_kwargs,
        #handle_parsing_errors=True,
    )

    print (type(agent))

    return cls(agent=agent, llm=llm, verbose=verbose)


def clean_log(s):
    s = re.sub(r".+?m", '', s)
    if re.search(r"\B> ", s):
        return None
    s = re.sub(
        r"\s(?P<name>Action Input|Action|Observation|Final Thought|Thought|Input|Final Answer|Answer|Property|Materials):", 
        lambda t: "\n`{}`:".format(t.group('name')), 
        s,
    )
    s = re.sub(r"\[(?P<name>.+?)\]", lambda t: '**[{}]**'.format(t.group('name')), s)
    return s.strip()


@contextlib.contextmanager
def capture_stdout():
    class Stream:
        def write(self, s):
            s = clean_log(s)
            if s:
                st.write(s)

    old_stdout = sys.stdout
    sys.stdout = Stream()  # Redirect stdout
    try:
        yield
    finally:
        sys.stdout = old_stdout  # Restore stdout