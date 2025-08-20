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
        max_iterations=5,
        guidance_criteria=guidance_criteria,
        retriever=retriever,
        runner_factory=runner_factory,
        user_id=_user_id,
        session_id=_session_id,
    )
    
    runner = await runner_factory.get_runner(agent=iterative_agent)
    part = Part(text=initial_query)
    content = Content(role="user", parts=[part])
    
    async for event in runner.run_async(
        user_id=_user_id,
        session_id=_session_id,
        new_message=content,
    ):
        pass

    best_query = runner.session.state.get("best_query", initial_query)
    output_text = runner.session.state.get("output_text", "")
    result = {"best_query": best_query, "output_text": output_text}

    print(f"\n\nFinal Result:\n{json.dumps(result, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
