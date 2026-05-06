from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None
    response_format: str | None = None
    model_id: str | None = None
    watchlist_id: str | None = None
    as_of_date: str | None = None


class QueryResponse(BaseModel):
    query: str
    routed_to: str
    reasoning: str
    response: str
    routing_confidence: float | None = None
    low_confidence: bool = False


class DirectQueryRequest(BaseModel):
    query: str
    session_id: str | None = None
    response_format: str | None = None
    model_id: str | None = None
    mode: str | None = None
    watchlist_id: str | None = None
    as_of_date: str | None = None


class DirectQueryResponse(BaseModel):
    agent_id: str
    query: str
    response: str
