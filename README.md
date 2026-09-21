# Deep Research, homelab edition

A homelab adaptation of [LangChain's Open Deep Research](https://github.com/langchain-ai/open_deep_research):
a LangGraph deep-research agent re-wired to run entirely on my own
inference stack. Upstream features (multiple model providers, search
APIs, MCP compatibility) still work; the defaults and the docs here
describe my setup.

> **Status (September 2026):** This repo captures the April 2026
> adaptation. Since then the serving stack moved to GLM-5.3-Flash and
> Qwen3.6-35B behind a LiteLLM gateway, and search became a
> multi-provider layer (Parallel/You.com/Brave/Tavily/SearXNG) in the
> agent harness. The MCP wrapper and adaptation approach still stand;
> model and endpoint specifics below are period-accurate. The code
> itself is now model-agnostic: defaults point at the
> `openai:research` alias resolved via `RESEARCH_BASE_URL`, so the
> gateway front-ends whatever is current.

## What changed from upstream

- Models point at a self-hosted OpenAI-compatible endpoint through the
  gateway alias (`research_model: openai:research`, resolved via
  `RESEARCH_BASE_URL`) — not OpenAI's API.
- Search runs through an MCP server (`SEARCH_MCP_URL`, the complete MCP
  endpoint URL); the April 2026 setup used a self-hosted SearXNG MCP
  server exposing a `search` tool (`search_api: searxng`), not Tavily.
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
  ├─ Supervisor (gateway alias) → splits research into sub-topics
  │    └─ Researcher sub-agents (gateway alias) → search via MCP
  └─ Synthesizer (gateway alias) → final report with citations
```

## Running it

Prerequisites: an OpenAI-compatible endpoint (LiteLLM gateway, SGLang,
or vLLM) serving any tool-calling model, and an MCP search server.

```bash
git clone https://github.com/mark-yong/deep-research.git
cd deep-research
uv venv && source .venv/bin/activate
uv sync

# point the agent at your own endpoints
export RESEARCH_BASE_URL=http://your-gateway:4000/v1
export SEARCH_MCP_URL=http://your-host:8080/mcp
```

The default model alias `openai:research` must exist at the gateway
(LiteLLM: a model_name in config; a raw SGLang/vLLM endpoint: its
served model id, e.g. `openai:glm-5.3-flash`). Override per-run via
the `research_model` / `summarization_model` / `final_answer_model`
config knobs if your alias differs.

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
| `research_model` | `openai:research` | Research sub-agent model (gateway alias) |
| `max_researcher_iterations` | `10` | Supervisor cycles |
| `max_concurrent_research_units` | `3` | Parallel sub-agents per cycle |
| `allow_clarification` | `false` | Skip clarification (programmatic use) |

## Attribution

Everything except the changes listed above is upstream
[open_deep_research](https://github.com/langchain-ai/open_deep_research)
(MIT, LangChain), by LangChain. Upstream README, benchmarks, and
docs remain in the git history; see
[src/legacy/](src/legacy/legacy.md) for the older single-graph variant.
