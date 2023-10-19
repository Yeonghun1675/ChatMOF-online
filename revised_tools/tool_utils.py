import warnings
from typing import Dict, Callable, List
from pydantic import ValidationError
from langchain.base_language import BaseLanguageModel
from langchain.tools.base import BaseTool
from langchain.agents import load_tools, get_all_tool_names

from revised_tools.predictor import _get_predict_properties
from chatmof.tools.search_csv import _get_search_csv
from chatmof.tools.genetic_algorithm import _get_generator
from chatmof.tools.visualizer import _get_visualizer
#from chatmof.tools.python_repl import _get_python_repl
from revised_tools.python_repl import _get_python_repl


_MOF_TOOLS: Dict[str, Callable[[BaseLanguageModel], BaseTool]] = {
    "search_csv" : _get_search_csv,
    "predictor": _get_predict_properties,
    'generator': _get_generator,
    "visualizer": _get_visualizer,
    "python_repl": _get_python_repl,
}

_load_tool_names: List[str] = [
    'llm-math',
]


def load_chatmof_tools_revised(
        llm: BaseLanguageModel, 
        verbose: bool=False,
        search_internet: bool=True,    
    ) -> List[BaseTool]:
    tools = load_tools(_load_tool_names, llm=llm)
    custom_tools = [model(llm=llm, verbose=verbose) for model in _MOF_TOOLS.values()]

    return custom_tools + tools