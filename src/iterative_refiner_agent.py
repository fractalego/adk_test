from typing import AsyncGenerator, override
from pydantic import Field
from google.adk.agents import BaseAgent, InvocationContext
from google.adk.events.event import Event
from google.genai.types import Part, Content
from src.agents_factory import AgentsFactory
from src.retriever import Retriever

_accepted_tag = "APPROVED!"
_modify_tag = "MODIFY_QUERY"


class IterativeRefinerAgent(BaseAgent):
    max_iterations: int = Field(default=5)
    guidance_criteria: list[str] = Field(default_factory=list)
    retriever: Retriever = Field(exclude=True)
    
    def __init__(
        self,
        name: str,
        max_iterations: int,
        guidance_criteria: list[str],
        retriever: Retriever,
    ):
        sequential_agent = AgentsFactory.create_agents(
            iteration=0,
            guidance_criteria=guidance_criteria,
            original_query="",
            accepted_tag=_accepted_tag,
            modify_tag=_modify_tag,
        )
        
        super().__init__(
            name=name, 
            sub_agents=[sequential_agent],
            max_iterations=max_iterations,
            guidance_criteria=guidance_criteria,
            retriever=retriever,
        )

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        initial_query = ctx.user_content.parts[0].text
        
        best_query = initial_query
        best_output = ""
        current_query = initial_query
        
        for iteration in range(self.max_iterations):
            print(f"\n=== Iteration {iteration + 1}: Query = '{current_query}' ===")

            results = self.retriever.get_chunks(current_query, top_k=8)
            print(f"Retrieved {len(results)} Chunks")
            formatted_chunks = self._format_chunks(results)

            sequential_agent = AgentsFactory.create_agents(
                iteration=iteration,
                guidance_criteria=self.guidance_criteria,
                original_query=current_query,
                accepted_tag=_accepted_tag,
                modify_tag=_modify_tag,
            )
            
            sequential_agent.sub_agents[0].set_chunks(formatted_chunks)
            
            part = Part(text=current_query)
            content = Content(role="user", parts=[part])
            
            sub_ctx = InvocationContext(
                artifact_service=ctx.artifact_service,
                session_service=ctx.session_service,
                memory_service=ctx.memory_service,
                credential_service=ctx.credential_service,
                invocation_id=ctx.invocation_id,
                branch=ctx.branch,
                agent=sequential_agent,
                user_content=content,
                session=ctx.session,
                run_config=ctx.run_config,
                plugin_manager=ctx.plugin_manager,
            )
            
            is_accepted = False

            async for event in sequential_agent.run_async(sub_ctx):
                output_text = event.content.parts[0].text

                if _accepted_tag in output_text:
                    print(f"\n=== APPROVED after {iteration + 1} iterations ===")
                    best_query = current_query
                    best_output = output_text
                    is_accepted = True
                    yield event
                    break

                elif _modify_tag in output_text:
                    current_query = output_text.split(f"{_modify_tag}:")[1].strip()
                    print(f"\n=== Query modified to: '{current_query}' ===")

                else:
                    best_output = output_text
                    yield event

            if is_accepted:
                break

        ctx.session.state["best_query"] = best_query
        ctx.session.state["output_text"] = best_output
        
        import json
        final_result = {"best_query": best_query, "output_text": best_output}
        final_content = Content(role="model", parts=[Part(text=json.dumps(final_result, indent=2))])
        final_event = Event(
            content=final_content,
            author=self.name,
            invocation_id=ctx.invocation_id,
            branch=ctx.branch
        )
        yield final_event

    def _format_chunks(self, chunks: list[tuple[str, str, float]]) -> str:
        formatted_chunks = []
        for i, (chunk, chunk_id, score) in enumerate(chunks, 1):
            formatted_chunks.append(
                f"Chunk {i} (ID: {chunk_id}, Score: {score:.3f}):\n{chunk}\n"
            )
        return "\n".join(formatted_chunks) if formatted_chunks else "No chunks retrieved"