# Deep Research, homelab edition

A homelab adaptation of [LangChain's Open Deep Research](https://github.com/langchain-ai/open_deep_research):
a LangGraph deep-research agent re-wired to run entirely on my own
inference stack. Upstream features (multiple model providers, search
APIs, MCP compatibility) still work; the defaults and the docs here
describe my setup.

## What changed from upstream

- Models point at a self-hosted SGLang endpoint serving MiniMax M2.7
  NVFP4 (`research_model: minimax:m27sg`), not OpenAI.
- Search runs through a self-hosted SearXNG MCP server
  (`search_api: searxng`), not Tavily.
- Added `src/open_deep_research/mcp_server.py` (~180 lines): exposes the
  agent itself as an MCP tool, so agent harnesses can call deep research
  directly.
- Supervisor budget raised for self-hosted use: `max_researcher_iterations`
  10, `max_concurrent_research_units` 3, clarification disabled for
  programmatic runs.
- Evaluation harness kept from upstream (pairwise + bench runners under
  `tests/`), plus experiment result JSONLs in `tests/expt_results/`.

## Architecture

```
agent harness → deep_research (MCP tool) → LangGraph agent
  ├─ Supervisor (m27sg) → splits research into sub-topics
  │    └─ Researcher sub-agents (m27sg) → search via SearXNG MCP
  └─ Synthesizer (m27sg) → final report with citations
```

## Running it

Prerequisites: a SGLang (or vLLM) endpoint serving any tool-calling
model, and a SearXNG MCP server.

```bash
git clone https://github.com/mark-yong/deep-research.git
cd deep-research
uv venv && source .venv/bin/activate
uv sync

# point the agent at your own endpoints
export MINIMAX_BASE_URL=http://your-host:PORT/v1
export SEARXNG_MCP_URL=http://your-host:PORT/mcp
```

Serve the agent as an MCP tool:

```bash
python -m open_deep_research.mcp_server
```

Or drive it through LangGraph Studio like upstream:

```bash
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

Key knobs live in `src/open_deep_research/configuration.py`:

| Setting | Default | Description |
|---|---|---|
| `search_api` | `searxng` | Search provider |
| `research_model` | `minimax:m27sg` | Research sub-agent model |
| `max_researcher_iterations` | `10` | Supervisor cycles |
| `max_concurrent_research_units` | `3` | Parallel sub-agents per cycle |
| `allow_clarification` | `false` | Skip clarification (programmatic use) |

## Attribution

Everything except the changes listed above is upstream
[open_deep_research](https://github.com/langchain-ai/open_deep_research)
(MIT, LangChain), by LangChain. Upstream README, benchmarks, and
docs remain in the git history; see
[src/legacy/](src/legacy/legacy.md) for the older single-graph variant.
