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
      "command": ["/home/myong/projects/deep-research/.venv/bin/python", "-m", "open_deep_research.mcp_server"],
      "enabled": true
    }
  }
}
"""

import asyncio
import json
import os
import sys
from typing import Any, AsyncIterator

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from open_deep_research.deep_researcher import deep_researcher
from open_deep_research.configuration import Configuration, SearchAPI

server = Server(
    "deep-research",
    version="0.1.0",
    instructions="Deep research agent: multi-source web research with synthesized reports and citations.",
)


def _build_runnable_config() -> dict:
    searxng_url = os.environ.get("SEARXNG_MCP_URL", "http://192.168.68.104:8080/mcp")
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
            "research_model": "minimax:m27sg",
            "summarization_model": "minimax:m27sg",
            "compression_model": "minimax:m27sg",
            "final_report_model": "minimax:m27sg",
        },
        "metadata": {"owner": "opencode"},
    }


def _emit_progress(phase: str, message: str) -> None:
    print(json.dumps({"type": "progress", "phase": phase, "message": message}), file=sys.stderr, flush=True)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="deep_research",
            description=(
                "Conduct deep research on any topic. Performs multi-pass web searching, "
                "source extraction, and synthesis into a comprehensive report with citations. "
                "Best for complex, multi-faceted research questions that require gathering "
                "information from many sources. Returns a structured report with inline "
                "citations and a sources section."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The research question or topic to investigate. "
                            "Be specific and include key dimensions or angles to explore."
                        ),
                    },
                },
                "required": ["query"],
            },
        )
    ]


async def _stream_research(query: str) -> AsyncIterator[dict[str, Any]]:
    config = _build_runnable_config()
    input_state = {"messages": [{"type": "human", "content": query}]}

    async for event in deep_researcher.astream(input_state, config):
        node = event.get("__node_name__") if isinstance(event, dict) else None

        if node == "clarify_with_user":
            yield {"phase": "briefing", "message": "Analyzing research scope..."}
        elif node == "write_research_brief":
            yield {"phase": "briefing", "message": "Research brief generated."}
        elif node == "research_supervisor":
            yield {"phase": "researching", "message": "Research in progress..."}
        elif node == "final_report_generation":
            yield {"phase": "synthesizing", "message": "Synthesizing final report..."}
        else:
            yield {"phase": "working", "message": "Processing..."}


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
        async for step in _stream_research(query):
            phase = step.get("phase", "unknown")
            message = step.get("message", "")
            progress_steps.append(f"[{phase.upper()}] {message}")
            _emit_progress(phase, message)
            final_state = step if isinstance(step, dict) else {}
    except Exception as e:
        _emit_progress("error", str(e))
        return [TextContent(type="text", text=json.dumps({"error": str(e), "query": query}, indent=2))]

    config = _build_runnable_config()
    input_state = {"messages": [{"type": "human", "content": query}]}

    try:
        final_state = await deep_researcher.ainvoke(input_state, config)
    except Exception as e:
        _emit_progress("error", str(e))
        return [TextContent(type="text", text=json.dumps({"error": str(e), "query": query}, indent=2))]

    final_report = final_state.get("final_report", "No report generated.")
    notes = final_state.get("notes", [])
    raw_notes = final_state.get("raw_notes", [])

    all_notes = notes + raw_notes
    if all_notes:
        sources_lines = []
        for i, note in enumerate(all_notes, 1):
            text = str(note)
            excerpt = text[:400] + "..." if len(text) > 400 else text
            sources_lines.append(f"[{i}] {excerpt}")
        final_report = final_report + "\n\n---\n\n### Sources\n\n" + "\n".join(sources_lines)

    result = {
        "query": query,
        "report": final_report,
        "steps": progress_steps,
        "sources_count": len(all_notes),
    }

    _emit_progress("done", f"Research complete. {len(all_notes)} sources gathered.")

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def main():
    await stdio_server(server)


if __name__ == "__main__":
    asyncio.run(main())
