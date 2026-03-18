import os

AGENT_URLS = {
    "financial-agent": os.getenv("FINANCIAL_AGENT_URL", "http://localhost:9001"),
    "research-agent": os.getenv("RESEARCH_AGENT_URL", "http://localhost:9002"),
}

ROUTER_PROVIDER = os.getenv("ROUTER_PROVIDER", "groq")
