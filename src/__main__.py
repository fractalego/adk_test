import asyncio
import os
import json

from src.iterations import iterate_agents
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

    result = await iterate_agents(
        initial_query=initial_query,
        max_iterations=5,
        guidance_criteria=guidance_criteria,
        retriever=retriever,
        runner_factory=runner_factory,
        user_id=_user_id,
        session_id=_session_id,
    )

    print(f"\n\nFinal Result:\n{json.dumps(result, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
