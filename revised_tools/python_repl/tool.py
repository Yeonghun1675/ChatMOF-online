from langchain.tools.base import BaseTool
from revised_tools.python_repl.base import PythonREPLTool


def _get_python_repl(**kwargs) -> BaseTool:
    return PythonREPLTool()