import asyncio
import os
import json

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from src.iterations import iterate_agents
from src.prompts.prompts import (
    retrieval_agent_instructions,
    generation_agent_instructions,
    refiner_agent_instructions,
)
from src.prompts.guidance_criteria import guidance_criteria
from src.tools import create_retrieval_tool
from src.runner_factory import RunnerFactory

initial_query = "Write the quarterly results"

_app_name = "test_app"
_user_id = "test_user"
_session_id = "test_session"


async def main():
    retrieval_tool = create_retrieval_tool(
        data_folder=os.path.join(os.path.dirname(__file__), "../data")
    )

    retrieval_agent = LlmAgent(
        model=LiteLlm(model="openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
        name="retrieval_agent",
        instruction=retrieval_agent_instructions,
        tools=[retrieval_tool],
    )

    generation_agent = LlmAgent(
        model=LiteLlm(model="openai/gpt-4o", api_key=os.getenv("OPENAI_API_KEY")),
        name="generation_agent",
        instruction=generation_agent_instructions,
    )

    criteria_text = "\n".join(f"- {criterion}" for criterion in guidance_criteria)
    refiner_agent = LlmAgent(
        model=LiteLlm(model="openai/gpt-4o", api_key=os.getenv("OPENAI_API_KEY")),
        name="refiner_agent",
        instruction=refiner_agent_instructions.format(criteria=criteria_text),
    )

    runner_factory = RunnerFactory(
        app_name=_app_name,
        user_id=_user_id,
        session_id=_session_id,
    )

    result = await iterate_agents(
        initial_query=initial_query,
        max_iterations=5,
        retrieval_agent=retrieval_agent,
        generation_agent=generation_agent,
        refiner_agent=refiner_agent,
        runner_factory=runner_factory,
        user_id=_user_id,
        session_id=_session_id,
    )

    print(f"\n\nFinal Result:\n{json.dumps(result, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
