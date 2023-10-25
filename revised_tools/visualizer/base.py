import os
import re
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List, Optional
from PIL import Image
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
import streamlit as st

import ase.io
import ase.visualize
from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManagerForChainRun

from chatmof.config import config
from chatmof.utils import search_file
from revised_tools.visualizer.prompt import PROMPT
from revised_tools.error import ChatMOFOnlineError


class Visualizer(Chain):
    """Tools that search csv using Pandas agent"""
    llm_chain: LLMChain
    data_dir: Path = Path(config['structure_dir'])
    input_key: str = 'question'
    output_key: str = 'answer'

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    def _write_log(self, action, text, run_manager):
        run_manager.on_text(f"\n[Visualizer] {action}: ", verbose=self.verbose)
        run_manager.on_text(text, verbose=self.verbose, color='yellow')

    def _parse_output(self, text:str) -> Dict[str, Any]:
        thought = re.search(r"(?<!Final )Thought:\s*(.+?)\s*(Material|Thought|Question|$)", text, re.DOTALL)
        material = re.search(r"Material:\s*(?:```|`)?(.+?)(?:```|`)?\s*(Material|Thought|Question|$)", text, re.DOTALL)

        if not material:
            raise ValueError(f'unknown format for LLM: {text}')
        
        return{
            'Thought': (thought.group(1) if thought else None),
            'Material': (material.group(1) if material else None),
        }

    def _visualize(self, f_st: Path) -> None:
        atoms = ase.io.read(f_st)
        ase.visualize.view(atoms)
    
    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None
    ):
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        
        llm_output = self.llm_chain.run(
            question = inputs[self.input_key],
            callbacks=callbacks,
            stop=['Question:'],
        )

        output = self._parse_output(llm_output)
        self._write_log('Thought', output['Thought'], run_manager)
        self._write_log('Material', output['Material'], run_manager)

        materials = output['Material'].split(',')
        for material in materials:
            material = material.strip()
            material = material.replace("_clean", "")

            if material in ['JUKPAI', 'XEGKUR', 'ROLEC19', 'PITPEP', 'LITDAV', 'TAHYUZ', 'MOJJUR']:
                cif = Path(f'cifs/{material}.cif').resolve()
                if not cif.exists():
                    raise FileNotFoundError(f'{cif} does not exists.')
                atoms = ase.io.read(cif)
                file = f'savefig/{material}.png'

                fig, ax = plt.subplots()
                plot_atoms(atoms, ax)#, radii=0.3, rotation=('45x,45y,45z'))
                plt.savefig(file, bbox_inches='tight')  # Save the figure
                plt.close(fig)  # Close the figure window
                image = Image.open(file)
                st.image(image, use_column_width=True,)

            else:
                raise ChatMOFOnlineError('ChatMOF online-demo does not support visualizer. If you want use more toolkits, please use code on our github.')

        return {self.output_key: f'The visualizer has successfully visualized the structure {materials}.'}
    
    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: str = PROMPT,
        **kwargs
    ) -> Chain:
        template = PromptTemplate(
            template=prompt,
            input_variables=['question'],
        )
        llm_chain = LLMChain(llm=llm, prompt=template)
        return cls(llm_chain=llm_chain, **kwargs)