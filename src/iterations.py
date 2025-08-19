from google.adk.agents import SequentialAgent, LlmAgent
from google.genai.types import Part, Content
from sqlalchemy.sql.functions import current_user

from src.runner_factory import RunnerFactory


async def iterate_agents(
    initial_query: str,
    max_iterations: int,
    retrieval_agent: LlmAgent,
    generation_agent: LlmAgent,
    refiner_agent: LlmAgent,
    runner_factory: RunnerFactory,
    user_id: str,
    session_id: str,
) -> dict:
    """
    Iteratively run agents to refine a query and generate output.

    Args:
        current_query: The initial query to start with.
        max_iterations: Maximum number of iterations to run.
        retrieval_agent: Agent responsible for retrieving relevant data.
        generation_agent: Agent responsible for generating content.
        refiner_agent: Agent responsible for refining the query.
        runner_factory: Factory to create runners for executing agents.
        user_id: User identifier for the session.
        session_id: Session identifier for the conversation.

    Returns:
        A dictionary containing the best query and the final output text.
    """

    current_query = initial_query
    final_output = ""

    for iteration in range(max_iterations):
        print(f"\n=== Iteration {iteration + 1}: Query = '{current_query}' ===")

        sequential_agent = SequentialAgent(
            sub_agents=[retrieval_agent, generation_agent, refiner_agent],
            name=f"iteration_{iteration + 1}",
        )
        runner = await runner_factory.get_runner(agent=sequential_agent)
        part = Part(text=current_query)
        content = Content(role="user", parts=[part])
        iteration_output = ""
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content,
        ):
            if event.content and event.content.parts:
                output_text = event.content.parts[0].text
                if output_text:
                    print(output_text, end="", flush=True)
                    iteration_output += output_text

            if event.is_final_response():
                break

        if "APPROVED:" in iteration_output:
            final_output = iteration_output.split("APPROVED:")[1].strip()
            print(f"\n=== APPROVED after {iteration + 1} iterations ===")
            break
        elif "MODIFY_QUERY:" in iteration_output:
            current_query = iteration_output.split("MODIFY_QUERY:")[1].strip()
            print(f"\n=== Query modified to: '{current_query}' ===")
        else:
            print(
                f"\n=== No clear directive from refiner in iteration {iteration + 1} ==="
            )
            break

    result = {"best_query": current_query, "output_text": final_output}
    return result
