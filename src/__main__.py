import asyncio
import os
import json

from google.adk.agents import LlmAgent, LoopAgent
from google.adk.models.lite_llm import LiteLlm
from google.genai.types import Content, Part

from src.prompts.prompts import (
    retrieval_agent_instructions,
    generation_agent_instructions, 
    refiner_agent_instructions
)
from src.prompts.guidance_criteria import guidance_criteria
from src.tools import create_retrieval_tool
from src.runner_factory import RunnerFactory

initial_query = "Write the quarterly results"

_app_name = "test_app"
_user_id = "test_user"
_session_id = "test_session"


async def main():
    retrieval_tool = create_retrieval_tool("data/")
    
    retrieval_agent = LlmAgent(
        model=LiteLlm(model="openai/gpt-4o", api_key=os.getenv("OPENAI_API_KEY")),
        name="retrieval_agent",
        instruction=retrieval_agent_instructions,
        tools=[retrieval_tool]
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
    
    loop_agent = LoopAgent(
        sub_agents=[retrieval_agent, generation_agent, refiner_agent],
        name="document_analysis_loop",
        max_iterations=5
    )

    part = Part(text=initial_query)
    content = Content(role='user', parts=[part])

    runner_factory = RunnerFactory(
        app_name=_app_name,
        user_id=_user_id,
        session_id=_session_id,
    )
    runner = await runner_factory.get_runner(agent=loop_agent)

    final_query = initial_query
    final_output = ""
    
    async for event in runner.run_async(
        user_id=_user_id,
        session_id=_session_id,
        new_message=content,
    ):
        if event.content and event.content.parts:
            output_text = event.content.parts[0].text
            if output_text:
                print(output_text, end='', flush=True)
                
                if "MODIFY_QUERY:" in output_text:
                    final_query = output_text.split("MODIFY_QUERY:")[1].strip()
                elif "APPROVED:" in output_text:
                    final_output = output_text.split("APPROVED:")[1].strip()
                    
        if event.is_final_response():
            break
    
    result = {
        "best_query": final_query,
        "output_text": final_output
    }
    
    print(f"\n\nFinal Result:\n{json.dumps(result, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
