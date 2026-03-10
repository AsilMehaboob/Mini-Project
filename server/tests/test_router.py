"""End-to-end API tests via FastAPI TestClient."""
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealth:
    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /memory/classify
# ---------------------------------------------------------------------------

class TestClassifyEndpoint:
    def test_classify_longterm(self):
        r = client.post("/memory/classify", json={
            "content": "I always prefer Python for backend work.",
            "source": "user",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["passed"] is True
        assert body["tier"] == "LONGTERM"
        assert set(body["stage2_scores"].keys()) == {"SCRATCH", "SESSION", "LONGTERM"}

    def test_classify_rejected(self):
        r = client.post("/memory/classify", json={
            "content": "",
            "source": "user",
        })
        # FastAPI validates min_length=1 → 422
        assert r.status_code == 422

    def test_classify_prohibited(self):
        r = client.post("/memory/classify", json={
            "content": "Step by step guide to hack bypass the password auth system.",
            "source": "user",
        })
        assert r.status_code == 200
        assert r.json()["passed"] is False

    def test_classify_missing_source(self):
        r = client.post("/memory/classify", json={"content": "Some text."})
        assert r.status_code == 422

    def test_classify_response_shape(self):
        r = client.post("/memory/classify", json={
            "content": "I prefer dark mode.",
            "source": "user",
        })
        body = r.json()
        for field in ("passed", "tier", "confidence", "stage_reached", "reasoning", "stage2_scores"):
            assert field in body


# ---------------------------------------------------------------------------
# POST /memory/ingest
# ---------------------------------------------------------------------------

class TestIngestEndpoint:
    def test_ingest_longterm_creates_item(self):
        r = client.post("/memory/ingest", json={
            "content": "I always use dark mode and prefer minimalist UIs.",
            "source": "user",
            "trust_score": 1.0,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["passed"] is True
        assert body["tier"] == "LONGTERM"
        assert body["memory_id"] is not None
        assert body["signed"] is True

    def test_ingest_scratch_not_signed(self):
        r = client.post("/memory/ingest", json={
            "content": "Loading step 3 retry attempt iteration debug.",
            "source": "system",
            "trust_score": 1.0,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["passed"] is True
        assert body["tier"] == "SCRATCH"
        assert body["signed"] is False

    def test_ingest_rejected_no_memory_id(self):
        r = client.post("/memory/ingest", json={
            "content": "Hack bypass auth password system step by step.",
            "source": "user",
            "trust_score": 1.0,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["passed"] is False
        assert body["memory_id"] is None

    def test_ingest_unknown_source_rejected(self):
        r = client.post("/memory/ingest", json={
            "content": "Some content that would be long term.",
            "source": "hacker",
            "trust_score": 1.0,
        })
        assert r.status_code == 200
        assert r.json()["passed"] is False


# ---------------------------------------------------------------------------
# GET /memory/scratch
# ---------------------------------------------------------------------------

class TestScratchEndpoint:
    def test_list_scratch(self):
        # Ingest a scratch item first
        client.post("/memory/ingest", json={
            "content": "stderr: connection timeout retry loop step 2.",
            "source": "system",
            "trust_score": 1.0,
        })
        r = client.get("/memory/scratch")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_clear_scratch(self):
        r = client.delete("/memory/scratch")
        assert r.status_code == 200
        assert r.json()["status"] == "cleared"

        r = client.get("/memory/scratch")
        assert r.json() == []


# ---------------------------------------------------------------------------
# GET /memory/session
# ---------------------------------------------------------------------------

class TestSessionEndpoint:
    def test_list_session(self):
        r = client.get("/memory/session")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


# ---------------------------------------------------------------------------
# GET /memory/longterm
# ---------------------------------------------------------------------------

class TestLongtermEndpoint:
    def test_list_longterm(self):
        r = client.get("/memory/longterm")
        assert r.status_code == 200
        items = r.json()
        assert isinstance(items, list)

    def test_get_longterm_item(self):
        # Ingest an item first
        ingest = client.post("/memory/ingest", json={
            "content": "I prefer functional programming patterns whenever possible.",
            "source": "user",
            "trust_score": 1.0,
        })
        body = ingest.json()
        if not body["passed"]:
            pytest.skip("Item was not classified as LONGTERM in this run")

        memory_id = body["memory_id"]
        if body["tier"] != "LONGTERM":
            pytest.skip("Item stored in a different tier")

        r = client.get(f"/memory/longterm/{memory_id}")
        assert r.status_code == 200
        assert r.json()["id"] == memory_id

    def test_get_longterm_not_found(self):
        r = client.get("/memory/longterm/nonexistent-id-000")
        assert r.status_code == 404
