import os

AGENT_URLS = {
    "financial-agent": "http://localhost:9001",
    "research-agent": "http://localhost:9002",
}

ROUTER_PROVIDER = os.getenv("ROUTER_PROVIDER", "groq")
