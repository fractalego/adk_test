from google.adk.agents import LlmAgent

from pydantic import Field

from src.prompts.prompts import generation_agent_instructions_template


class GenerationAgent(LlmAgent):
    chunks_state: str = Field(default="")
    
    def set_chunks(self, chunks: str):
        self.chunks_state = chunks
        base_instruction = generation_agent_instructions_template
        self.instruction = base_instruction.format(chunks=chunks)
