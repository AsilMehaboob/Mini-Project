import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .classifier.pipeline import run as pipeline_run, ClassificationResult
from .classifier.stage_4.policy import DEFAULT_ALLOWED_SOURCES
from .crypto import generate_keypair, sign_item
from .longterm import LongTermMemory
from .models import MemoryItem
from .scratch import ScratchMemory
from .session import SessionMemory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])

# ---------------------------------------------------------------------------
# In-process singletons (replaced by proper DI / persistence layer later)
# ---------------------------------------------------------------------------

_private_key, _public_key = generate_keypair()
_scratch  = ScratchMemory()
_session  = SessionMemory()
_longterm = LongTermMemory(_public_key)

SESSION_TTL_HOURS = 24

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ClassifyRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10_000)
    source:  str = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Pipeline tuning
    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    min_trust_score:      float = Field(default=0.0,  ge=0.0, le=1.0)
    allowed_sources: Optional[List[str]] = Field(default=None)

    # Stage 5 LLM config (optional)
    llm_api_key:  Optional[str] = Field(default=None)
    llm_base_url: str = Field(default="https://api.openai.com/v1")
    llm_model:    str = Field(default="gpt-4o-mini")


class IngestRequest(ClassifyRequest):
    trust_score: float = Field(default=1.0, ge=0.0, le=1.0)


class ClassifyResponse(BaseModel):
    passed:        bool
    tier:          Optional[str]
    confidence:    float
    stage_reached: int
    reasoning:     str
    stage2_scores: Dict[str, float]


class IngestResponse(ClassifyResponse):
    memory_id: Optional[str]
    signed:    bool


class MemoryItemResponse(BaseModel):
    id:          str
    content:     str
    source:      str
    tier:        str
    created_at:  datetime
    trust_score: float
    signature:   Optional[str]
    expires_at:  Optional[datetime]

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_metadata(req: ClassifyRequest) -> Dict[str, Any]:
    meta = {"source": req.source, **req.metadata}
    if isinstance(req, IngestRequest):
        meta["trust_score"] = req.trust_score
    return meta


def _run_pipeline(req: ClassifyRequest) -> ClassificationResult:
    return pipeline_run(
        content=req.content,
        metadata=_build_metadata(req),
        confidence_threshold=req.confidence_threshold,
        allowed_sources=set(req.allowed_sources) if req.allowed_sources else None,
        min_trust_score=req.min_trust_score,
        llm_api_key=req.llm_api_key,
        llm_base_url=req.llm_base_url,
        llm_model=req.llm_model,
    )


def _to_item_response(item: MemoryItem) -> MemoryItemResponse:
    return MemoryItemResponse(
        id=item.id,
        content=item.content,
        source=item.source,
        tier=item.tier,
        created_at=item.created_at,
        trust_score=item.trust_score,
        signature=item.signature,
        expires_at=item.expires_at,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/classify", response_model=ClassifyResponse, summary="Classify a memory candidate")
def classify(req: ClassifyRequest) -> ClassifyResponse:
    """
    Run the full classification pipeline (Stages 1–5) on a text candidate
    without storing anything. Useful for dry-runs and testing.
    """
    result = _run_pipeline(req)
    return ClassifyResponse(
        passed=result.passed,
        tier=result.tier,
        confidence=result.confidence,
        stage_reached=result.stage_reached,
        reasoning=result.reasoning,
        stage2_scores=result.stage2_scores,
    )


@router.post("/ingest", response_model=IngestResponse, summary="Classify and store a memory")
def ingest(req: IngestRequest) -> IngestResponse:
    """
    Classify a candidate memory and, if accepted, store it in the appropriate
    tier (SCRATCH → ScratchMemory, SESSION → SessionMemory,
    LONGTERM → signed + LongTermMemory).
    """
    result = _run_pipeline(req)

    if not result.passed or result.tier is None:
        return IngestResponse(
            passed=False,
            tier=result.tier,
            confidence=result.confidence,
            stage_reached=result.stage_reached,
            reasoning=result.reasoning,
            stage2_scores=result.stage2_scores,
            memory_id=None,
            signed=False,
        )

    tier = result.tier
    expires_at = (
        datetime.now(timezone.utc) + timedelta(hours=SESSION_TTL_HOURS)
        if tier == "SESSION"
        else None
    )

    item = MemoryItem.create(
        content=req.content,
        source=req.source,
        tier=tier,
        trust_score=req.trust_score,
        expires_at=expires_at,
    )

    signed = False
    if tier == "LONGTERM":
        item = sign_item(item, _private_key)
        signed = True

    try:
        if tier == "SCRATCH":
            _scratch.add(item)
        elif tier == "SESSION":
            _session.add(item)
        elif tier == "LONGTERM":
            _longterm.add(item)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    logger.info("Ingest | stored id=%s tier=%s signed=%s", item.id, tier, signed)

    return IngestResponse(
        passed=True,
        tier=tier,
        confidence=result.confidence,
        stage_reached=result.stage_reached,
        reasoning=result.reasoning,
        stage2_scores=result.stage2_scores,
        memory_id=item.id,
        signed=signed,
    )


# ── Scratch ──────────────────────────────────────────────────────────────────

@router.get(
    "/scratch",
    response_model=List[MemoryItemResponse],
    summary="List all scratch (ephemeral) memories",
)
def list_scratch() -> List[MemoryItemResponse]:
    return [_to_item_response(i) for i in _scratch.get_all()]


@router.delete("/scratch", summary="Discard all scratch memories")
def clear_scratch() -> Dict[str, str]:
    _scratch.clear()
    return {"status": "cleared"}


# ── Session ───────────────────────────────────────────────────────────────────

@router.get(
    "/session",
    response_model=List[MemoryItemResponse],
    summary="List active (non-expired) session memories",
)
def list_session() -> List[MemoryItemResponse]:
    _session.purge_expired()
    return [_to_item_response(i) for i in _session.get_active()]


# ── Long-term ─────────────────────────────────────────────────────────────────

@router.get(
    "/longterm",
    response_model=List[MemoryItemResponse],
    summary="List all verified long-term memories",
)
def list_longterm() -> List[MemoryItemResponse]:
    return [_to_item_response(i) for i in _longterm.get_all_verified()]


@router.get(
    "/longterm/{memory_id}",
    response_model=MemoryItemResponse,
    summary="Retrieve and verify a single long-term memory",
)
def get_longterm(memory_id: str) -> MemoryItemResponse:
    try:
        item = _longterm.get(memory_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id!r} not found")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _to_item_response(item)
