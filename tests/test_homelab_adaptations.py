"""Regression tests for the homelab adaptations.

These cover the parts this fork changed from upstream LangChain:
gateway-aliased models, MCP search plumbing, and the agent-as-MCP-tool
wrapper. They are offline: no network, no model calls.
"""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("RESEARCH_BASE_URL", "http://test-gateway:4000/v1")

from open_deep_research.configuration import Configuration, SearchAPI  # noqa: E402
from open_deep_research.utils import load_mcp_tools  # noqa: E402


# --------------------------------------------------------------------------
# 1. SEARCH_MCP_URL semantics: the configured URL is used verbatim, no
#    '/mcp' appended (the old code produced .../mcp/mcp).
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_url_used_verbatim():
    """mcp_config.url is the complete endpoint URL and is passed through
    to MultiServerMCPClient unchanged (minus a trailing slash)."""
    captured = {}

    class FakeClient:
        def __init__(self, cfg):
            captured["cfg"] = cfg

        async def get_tools(self):
            tool = MagicMock()
            tool.name = "search"
            return [tool]

    config = {
        "configurable": {
            "search_api": SearchAPI.SEARXNG.value,
            "mcp_config": {
                "url": "http://host:8080/mcp",
                "tools": ["search"],
                "auth_required": False,
            },
        }
    }

    with patch("open_deep_research.utils.MultiServerMCPClient", FakeClient):
        tools = await load_mcp_tools(config, existing_tool_names=set())

    server_cfg = captured["cfg"]["server_1"]
    assert server_cfg["url"] == "http://host:8080/mcp", (
        f"endpoint mutated: {server_cfg['url']}"
    )
    assert len(tools) == 1


@pytest.mark.asyncio
async def test_mcp_url_trailing_slash_normalized_not_extended():
    config = {
        "configurable": {
            "mcp_config": {
                "url": "http://host:8080/mcp/",
                "tools": ["search"],
                "auth_required": False,
            },
        }
    }
    captured = {}

    class FakeClient:
        def __init__(self, cfg):
            captured["cfg"] = cfg

        async def get_tools(self):
            tool = MagicMock()
            tool.name = "search"
            return [tool]

    with patch("open_deep_research.utils.MultiServerMCPClient", FakeClient):
        await load_mcp_tools(config, existing_tool_names=set())

    assert captured["cfg"]["server_1"]["url"] == "http://host:8080/mcp"


# --------------------------------------------------------------------------
# 2. MCP failures fail loudly when search_api=searxng depends on them.
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_connection_failure_raises():
    """An unreachable search backend must raise, not silently return []."""

    class BoomClient:
        def __init__(self, cfg):
            pass

        async def get_tools(self):
            raise ConnectionError("connection refused")

    config = {
        "configurable": {
            "mcp_config": {
                "url": "http://host:8080/mcp",
                "tools": ["search"],
                "auth_required": False,
            },
        }
    }
    with patch("open_deep_research.utils.MultiServerMCPClient", BoomClient):
        with pytest.raises(RuntimeError, match="Failed to load MCP tools"):
            await load_mcp_tools(config, existing_tool_names=set())


@pytest.mark.asyncio
async def test_mcp_missing_requested_tool_raises():
    """Server reachable but not exposing the requested tool names -> raise,
    listing what IS available."""

    class OkClient:
        def __init__(self, cfg):
            pass

        async def get_tools(self):
            tool = MagicMock()
            tool.name = "some_other_tool"
            return [tool]

    config = {
        "configurable": {
            "mcp_config": {
                "url": "http://host:8080/mcp",
                "tools": ["search"],
                "auth_required": False,
            },
        }
    }
    with patch("open_deep_research.utils.MultiServerMCPClient", OkClient):
        with pytest.raises(RuntimeError, match="some_other_tool"):
            await load_mcp_tools(config, existing_tool_names=set())


# --------------------------------------------------------------------------
# 3. One deep_research MCP invocation runs the graph exactly once.
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_call_tool_runs_graph_once():
    """The wrapper must execute the research graph exactly once per call:
    progress comes from the same astream() that produces the final state."""
    import open_deep_research.mcp_server as mcp_server

    calls = {"astream": 0, "ainvoke": 0}

    async def fake_astream(*args, **kwargs):
        calls["astream"] += 1
        # values events carry full state; the last one is the terminal state
        yield ((), "values", {"messages": [], "final_report": ""})
        yield ((), "updates", {"clarify_with_user": {"messages": []}})
        yield ((), "updates", {"write_research_brief": {"research_brief": "b"}})
        yield ((), "updates", {"research_supervisor": {"notes": ["n"]}})
        yield ((), "updates", {"final_report_generation": {"final_report": "R"}})
        yield ((), "values", {"messages": [], "final_report": "R", "notes": ["n"]})

    with patch.object(mcp_server.deep_researcher, "astream", side_effect=fake_astream), \
         patch.object(mcp_server.deep_researcher, "ainvoke",
                      side_effect=AssertionError("ainvoke must not be called")):
        result = await mcp_server.call_tool("deep_research", {"query": "test query"})

    assert calls["astream"] == 1
    payload = json.loads(result[0].text)
    assert payload["report"] == "R"
    # progress phases derived from real node names
    assert any("BRIEFING" in s for s in payload["steps"])
    assert any("RESEARCHING" in s for s in payload["steps"])
    assert any("SYNTHESIZING" in s for s in payload["steps"])
    # no fabricated sources section / no note-blob source count
    assert "### Sources" not in payload["report"]
    assert "sources_count" not in payload


@pytest.mark.asyncio
async def test_mcp_call_tool_rejects_unknown_and_empty():
    import open_deep_research.mcp_server as mcp_server

    result = await mcp_server.call_tool("nope", {"query": "x"})
    assert "Unknown tool" in result[0].text

    result = await mcp_server.call_tool("deep_research", {"query": "  "})
    assert "query is required" in result[0].text


# --------------------------------------------------------------------------
# 4. Console entry point: main() is a sync wrapper, module runs standalone.
# --------------------------------------------------------------------------

def test_entry_point_is_synchronous():
    """pyproject points the console script at mcp_server:main; that must be
    a plain callable, not a coroutine function."""
    import inspect
    import open_deep_research.mcp_server as mcp_server

    assert not inspect.iscoroutinefunction(mcp_server.main)
    # and _amain exists as the awaitable implementation
    assert inspect.iscoroutinefunction(mcp_server._amain)


# --------------------------------------------------------------------------
# 5. Supervisor exception handling: only token limits end research early.
# --------------------------------------------------------------------------

def test_supervisor_reraises_non_token_limit_errors():
    """Inherited-upstream regression: the old 'or True' swallowed every
    exception. Non-token-limit errors must now propagate."""
    import ast as _ast

    src = open(
        os.path.join(os.path.dirname(__file__), "..", "src", "open_deep_research", "deep_researcher.py")
    ).read()
    tree = _ast.parse(src)
    for node in _ast.walk(tree):
        if isinstance(node, _ast.BoolOp) and isinstance(node.op, _ast.Or):
            values = [_ast.dump(v) for v in node.values]
            assert not any("or True" in v.lower() for v in values), (
                "found an 'or True' boolean shortcut in deep_researcher.py"
            )


# --------------------------------------------------------------------------
# 6. Gateway plumbing: RESEARCH_BASE_URL -> OPENAI_API_BASE, alias default.
# --------------------------------------------------------------------------

def test_gateway_env_forwarding():
    import importlib
    import open_deep_research.mcp_server as mcp_server

    importlib.reload(mcp_server)
    assert os.environ.get("OPENAI_API_BASE") == "http://test-gateway:4000/v1"


def test_default_model_alias():
    config = Configuration.from_runnable_config(None)
    assert config.research_model == "openai:research"
    assert config.summarization_model == "openai:research"
    assert config.compression_model == "openai:research"
    assert config.final_report_model == "openai:research"


def test_mcp_server_wrapper_uses_search_tool_contract():
    """The MCP wrapper must request a tool named 'search' — documented
    requirement for the search backend."""
    import open_deep_research.mcp_server as mcp_server

    cfg = mcp_server._build_runnable_config()["configurable"]
    assert cfg["mcp_config"]["tools"] == ["search"]
    assert cfg["search_api"] == SearchAPI.SEARXNG.value
