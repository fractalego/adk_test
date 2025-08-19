import asyncio
import os

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.genai.types import Content, Part

from src.prompts.prompts import generation_instructions
from src.runner_factory import RunnerFactory

initial_query = "do the summary."

_app_name = "test_app"
_user_id = "test_user"
_session_id = "test_session"


async def main():
    summariser_openai = LlmAgent(
        model=LiteLlm(model="openai/gpt-4o", api_key=os.getenv("OPENAI_API_KEY")),
        name="openai_agent",
        instruction=generation_instructions,
    )

    agent = SequentialAgent(
        sub_agents=[summariser_openai],
        name="sequential_agent",
    )

    part = Part(text=initial_query)
    content = Content(role='user', parts=[part])

    runner_factory = RunnerFactory(
        app_name=_app_name,
        user_id=_user_id,
        session_id=_session_id,
        initial_state={
            "text": "the quick brown fox jumps over the lazy dog. "
        }
    )
    runner = await runner_factory.get_runner(agent=agent)

    async for event in runner.run_async(
        user_id=_user_id,
        session_id=_session_id,
        new_message=content,
    ):
        print(event.content.parts[0].text, end='', flush=True)
        if event.is_final_response():
            print()



if __name__ == "__main__":
    asyncio.run(main())
