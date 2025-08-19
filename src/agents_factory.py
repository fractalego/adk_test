import os

from google.adk.agents import LlmAgent
from google.adk.agents import SequentialAgent
from google.adk.models.lite_llm import LiteLlm

from src.prompts.prompts import (
    generation_agent_instructions,
    refiner_agent_instructions,
)

_model_name = "openai/gpt-4o-mini"

class AgentsFactory:
    @staticmethod
    def create_agents(
        iteration: int, guidance_criteria: list[str], original_query: str, accepted_tag: str, modify_tag: str
    ) -> SequentialAgent:
        generation_agent = LlmAgent(
            model=LiteLlm(model=_model_name, api_key=os.getenv("OPENAI_API_KEY")),
            name="generation_agent",
            instruction=generation_agent_instructions,
        )

        criteria_text = "\n".join(f"- {criterion}" for criterion in guidance_criteria)
        refiner_agent = LlmAgent(
            model=LiteLlm(model=_model_name, api_key=os.getenv("OPENAI_API_KEY")),
            name="refiner_agent",
            instruction=refiner_agent_instructions.format(
                criteria=criteria_text,
                original_query=original_query,
                accepted_tag=accepted_tag,
                modify_tag=modify_tag
            ),
        )
        sequential_agent = SequentialAgent(
            sub_agents=[generation_agent, refiner_agent],
            name=f"iteration_{iteration + 1}",
        )
        return sequential_agent
