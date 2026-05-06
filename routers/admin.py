from fastapi import APIRouter

from app import registry

router = APIRouter(tags=["admin"])


@router.get("/health")
async def health():
    return {"status": "ok", "service": "agent-marketplace", "agents": len(registry.get_cards())}
