"""MCP server exposing deep_research as a tool with streaming progress.

Usage:
    python -m open_deep_research.mcp_server

Or install as entry point (after pip install -e .):
    deep-research-mcp

OpenCode config (~/.config/opencode/opencode.json):
{
  "mcp": {
    "deep_research": {
      "type": "local",
      "command": ["python", "-m", "open_deep_research.mcp_server"],
      "enabled": true
    }
  }
}
"""

import asyncio
import json
import os
import sys
from typing import Any, AsyncIterator, cast

from langchain_core.runnables import RunnableConfig
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# RESEARCH_BASE_URL (OpenAI-compatible gateway) -> the OPENAI_API_BASE
# env var that init_chat_model/ChatOpenAI reads, before any client is
# constructed.
if os.environ.get("RESEARCH_BASE_URL"):
    os.environ.setdefault("OPENAI_API_BASE", os.environ["RESEARCH_BASE_URL"])

from open_deep_research.deep_researcher import deep_researcher
from open_deep_research.configuration import Configuration, SearchAPI

server = Server(
    "deep-research",
    version="0.1.0",
    instructions="Deep research agent: multi-source web research with synthesized reports and citations.",
)


def _build_runnable_config() -> dict:
    searxng_url = os.environ.get("SEARCH_MCP_URL", "http://localhost:8080/mcp")
    return {
        "configurable": {
            "search_api": SearchAPI.SEARXNG.value,
            "mcp_config": {"url": searxng_url, "tools": ["search"], "auth_required": False},
            "mcp_prompt": (
                "You have access to a SearXNG web search via MCP with a 'search' tool. "
                "Each call takes a query string and returns relevant web results "
                "with titles, URLs, and snippets. Use it to find information from the web."
            ),
            "allow_clarification": False,
            "max_researcher_iterations": 10,
            "max_concurrent_research_units": 3,
            "research_model": "openai:research",
            "summarization_model": "openai:research",
            "compression_model": "openai:research",
            "final_report_model": "openai:research",
        },
        "metadata": {"owner": "opencode"},
    }


def _emit_progress(phase: str, message: str) -> None:
    print(json.dumps({"type": "progress", "phase": phase, "message": message}), file=sys.stderr, flush=True)


# Human-readable phase labels for graph node names (langgraph streams
# updates as {node_name: update} dicts, optionally namespaced under
# subgraphs like ("research_supervisor:<uuid>", ...)).
_NODE_PHASES = [
    ("clarify_with_user", ("briefing", "Analyzing research scope...")),
    ("write_research_brief", ("briefing", "Research brief generated.")),
    ("research_supervisor", ("researching", "Research in progress...")),
    ("researcher", ("researching", "Research in progress...")),
    ("final_report_generation", ("synthesizing", "Synthesizing final report...")),
]


def _phase_for_node(node_name: str) -> tuple[str, str]:
    for prefix, phase in _NODE_PHASES:
        if node_name.startswith(prefix) or prefix in node_name:
            return phase
    return ("working", "Processing...")


async def _run_research_once(query: str) -> AsyncIterator[dict[str, Any]]:
    """Execute the graph exactly once, streaming progress while capturing
    the final state.

    The stream combines two modes:
      - "updates" (with subgraphs) yields (namespace, {node: update})
        tuples, used for progress reporting only;
      - "values" yields the full accumulated state; the final one is the
        graph's terminal state, captured for the result.

    The research graph runs exactly once per call: astream() both
    executes it and hands us the final state. Do not ainvoke() the same
    query afterwards — that would run the entire research a second time.
    """
    config: RunnableConfig = cast(Any, _build_runnable_config())
    input_state: dict[str, Any] = {"messages": [{"type": "human", "content": query}]}

    last_state: dict[str, Any] = {}
    async for chunk in deep_researcher.astream(
        input_state, config, stream_mode=["updates", "values"], subgraphs=True
    ):
        namespace, mode, payload = chunk
        if mode == "values":
            last_state = cast(dict[str, Any], payload)
            continue
        # updates mode: payload is {node_name: update}
        if not isinstance(payload, dict):
            continue
        for node_name in payload:
            phase, message = _phase_for_node(node_name)
            yield {"phase": phase, "message": message}

    yield {"phase": "final", "state": last_state}


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name != "deep_research":
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    query = arguments.get("query", "").strip()
    if not query:
        return [TextContent(type="text", text="Error: query is required")]

    _emit_progress("starting", f"Starting deep research: {query[:60]}...")

    progress_steps: list[str] = []
    final_state: dict[str, Any] = {}

    try:
        async for step in _run_research_once(query):
            if step.get("phase") == "final":
                final_state = step.get("state") or {}
                continue
            phase = step.get("phase", "unknown")
            message = step.get("message", "")
            progress_steps.append(f"[{phase.upper()}] {message}")
            _emit_progress(phase, message)
    except Exception as e:
        _emit_progress("error", str(e))
        return [TextContent(type="text", text=json.dumps({"error": str(e), "query": query}, indent=2))]

    final_report = final_state.get("final_report", "No report generated.")

    result = {
        "query": query,
        "report": final_report,
        "steps": progress_steps,
        "sources_count": len(final_state.get("notes", [])) or None,
    }
    if result["sources_count"] is None:
        del result["sources_count"]

    _emit_progress("done", "Research complete.")

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _amain() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """Synchronous entry point for the console script and `python -m`."""
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
