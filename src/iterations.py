from google.genai.types import Part, Content
from src.agents_factory import AgentsFactory
from src.runner_factory import RunnerFactory
from src.retriever import Retriever

_accepted_tag = "APPROVED!"
_modify_tag = "MODIFY_QUERY"


async def iterate_agents(
    initial_query: str,
    max_iterations: int,
    guidance_criteria: list[str],
    retriever: Retriever,
    runner_factory: RunnerFactory,
    user_id: str,
    session_id: str,
) -> dict:
    """
    Iteratively run agents to refine a query and generate output.

    Args:
        initial_query (str): The initial query to start with.
        max_iterations (int): The maximum number of iterations to run.
        guidance_criteria (list[str]): List of criteria to guide the agents.
        retriever (Retriever): The retriever instance to fetch document chunks.
        runner_factory (RunnerFactory): The runner factory instance to use.
        user_id (str): The user ID for the session.
        session_id (str): The session ID for the session.

    Returns:
        dict: A dictionary containing the best query and the final output text.
    """

    best_query = initial_query
    best_output = ""

    current_query = initial_query
    for iteration in range(max_iterations):
        print(f"\n=== Iteration {iteration + 1}: Query = '{current_query}' ===")

        results = retriever.get_chunks(current_query, top_k=8)
        print(f"Retrieved {len(results)} Chunks")
        formatted_chunks = _format_chunks(results)
        query_with_chunks = f"{current_query}\n\nRETRIEVED CHUNKS:\n{formatted_chunks}"

        sequential_agent = AgentsFactory.create_agents(
            iteration=iteration,
            guidance_criteria=guidance_criteria,
            original_query=current_query,
            accepted_tag=_accepted_tag,
            modify_tag=_modify_tag,
        )
        runner = await runner_factory.get_runner(agent=sequential_agent)
        part = Part(text=query_with_chunks)
        content = Content(role="user", parts=[part])
        best_output = ""
        is_accepted = False

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content,
        ):
            output_text = event.content.parts[0].text

            if _accepted_tag in output_text:
                print(f"\n=== APPROVED after {iteration + 1} iterations ===")
                best_query = current_query
                is_accepted = True
                break

            elif _modify_tag in output_text:
                current_query = output_text.split(f"{_modify_tag}:")[1].strip()
                print(f"\n=== Query modified to: '{current_query}' ===")

            else:
                best_output = output_text

        if is_accepted:
            break

    result = {"best_query": best_query, "output_text": best_output}
    return result


def _format_chunks(chunks: list[tuple[str, str, float]]) -> str:
    """
    Format the retrieved chunks into a readable string.

    Args:
        chunks (list[tuple[str, str, float]]): List of tuples containing chunk text, ID, and score.

    Returns:
        str: Formatted string of chunks.
    """
    formatted_chunks = []
    for i, (chunk, chunk_id, score) in enumerate(chunks, 1):
        formatted_chunks.append(
            f"Chunk {i} (ID: {chunk_id}, Score: {score:.3f}):\n{chunk}\n"
        )
    return "\n".join(formatted_chunks) if formatted_chunks else "No chunks retrieved"
