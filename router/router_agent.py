import logging

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import ROUTER_PROVIDER

logger = logging.getLogger("marketplace.router")


class RoutingDecision(BaseModel):
    """Structured output for routing decisions."""
    agent_name: str = Field(description="The agent_id to route the query to")
    reasoning: str = Field(description="Brief explanation of why this agent was chosen")


class RouterAgent:
    """Uses an LLM with structured output to decide which agent should handle a query."""

    def __init__(self, provider: str = ROUTER_PROVIDER):
        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            self._llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        else:
            from langchain_groq import ChatGroq
            self._llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

        self._structured_llm = self._llm.with_structured_output(RoutingDecision)

    async def route(self, query: str, routing_context: str) -> RoutingDecision:
        """Decide which agent should handle the query."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a query router. Given a user query, decide which specialist agent "
                "should handle it. Choose the most appropriate agent based on the query content "
                "and the agents' descriptions and skills.\n\n"
                "Available agents:\n{routing_context}\n\n"
                "Respond with the agent_id and your reasoning."
            )),
            ("human", "{query}"),
        ])

        chain = prompt | self._structured_llm
        decision = await chain.ainvoke({"query": query, "routing_context": routing_context})
        logger.info("Routing decision: agent='%s', reason='%s'", decision.agent_name, decision.reasoning)
        return decision
