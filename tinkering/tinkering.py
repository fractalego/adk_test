import asyncio
import os

from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

initial_query = "Hello! Can you tell me a joke?"

_app_name = "test_app"
_user_id = "test_user"
_session_id = "test_session"


async def main():
    agent_openai = LlmAgent(
        model=LiteLlm(model="openai/gpt-4o", api_key=os.getenv("OPENAI_API_KEY")),
        name="openai_agent",
        instruction="You are a helpful assistant ",
    )
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=_app_name,
        user_id=_user_id,
        session_id=_session_id,
    )
    part = Part(text=initial_query)
    content = Content(role="user", parts=[part])
    runner = Runner(
        agent=agent_openai, app_name=_app_name, session_service=session_service
    )
    async for event in runner.run_async(
        user_id=_user_id, session_id=_session_id, new_message=content
    ):
        print(event.content.parts[0].text, end="", flush=True)
        if event.is_final_response():
            print()


if __name__ == "__main__":
    asyncio.run(main())
