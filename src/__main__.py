import asyncio
import os
import json

from google.genai.types import Part, Content
from src.iterative_refiner_agent import IterativeRefinerAgent
from src.prompts.guidance_criteria import guidance_criteria
from src.retriever import Retriever
from src.runner_factory import RunnerFactory

initial_query = "Write the quarterly results"

_app_name = "test_app"
_user_id = "test_user"
_session_id = "test_session"


async def main():
    data_folder = os.path.join(os.path.dirname(__file__), "../data")
    retriever = Retriever(data_folder)
    runner_factory = RunnerFactory(
        app_name=_app_name,
        user_id=_user_id,
        session_id=_session_id,
    )

    iterative_agent = IterativeRefinerAgent(
        name="iterative_refiner",
        max_iterations=3,
        guidance_criteria=guidance_criteria,
        retriever=retriever,
    )
    
    runner = await runner_factory.get_runner(agent=iterative_agent)
    part = Part(text=initial_query)
    content = Content(role="user", parts=[part])
    
    result = {"best_query": initial_query, "output_text": ""}
    async for event in runner.run_async(
        user_id=_user_id,
        session_id=_session_id,
        new_message=content,
    ):
        if hasattr(event.content, 'parts') and event.content.parts:
            text = event.content.parts[0].text
            try:
                result = json.loads(text)
            except json.JSONDecodeError:
                pass

    print(f"\n\nFinal Result:\n{json.dumps(result, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
