# agent-marketplace

Router/gateway that discovers specialist agents via the A2A protocol and delegates user queries to the best agent using LLM-based routing.

## What It Does

- Fetches Agent Cards from registered A2A agents on startup
- Uses an LLM (Groq or Gemini) with structured output to route queries to the most appropriate agent
- Calls agents via A2A protocol (`message/send` JSON-RPC)
- Supports direct agent calls bypassing the router
- No frontend — API only

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/query` | LLM routes the query to the best agent, returns response + routing info |
| POST | `/agents/{agent_id}/query` | Call a specific agent directly via A2A (bypass router) |
| GET | `/agents` | List all registered agents with their Agent Cards |
| POST | `/agents/refresh` | Re-fetch Agent Cards from all agents |
| GET | `/health` | Health check |

## Registered Agents

Hardcoded in `config.py`:

| Agent ID | URL |
|----------|-----|
| `financial-agent` | `http://localhost:9001` |
| `research-agent` | `http://localhost:9002` |

## Structure

```
agent-marketplace/
├── app.py              # FastAPI gateway (port 9000)
├── config.py           # Agent URLs, router provider
├── pyproject.toml
└── router/
    ├── registry.py     # Agent Card fetching + caching
    ├── router_agent.py # LLM routing with structured output
    └── a2a_caller.py   # A2A protocol caller
```

## Prerequisites

Agent servers must be running:
- agent-financials A2A on port 9001
- agent-research A2A on port 9002

## Running

```bash
infisical run -- uvicorn app:app --host 0.0.0.0 --port 9000
```

## Testing

```bash
# Routed query (financial)
curl -X POST http://localhost:9000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze RELIANCE.NS stock"}'

# Routed query (research)
curl -X POST http://localhost:9000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Find papers on transformer architectures"}'

# List agents
curl http://localhost:9000/agents
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key (default router provider) |
| `GOOGLE_API_KEY` | Google AI API key (if using gemini provider) |
| `ROUTER_PROVIDER` | `groq` (default) or `gemini` |

## Dependencies

`fastapi`, `uvicorn`, `a2a-sdk`, `httpx`, `pydantic`, `python-dotenv`, `langchain-groq`, `langchain-google-genai`, `langchain`
