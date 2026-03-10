import logging

from fastapi import FastAPI

from memory.router import router as memory_router

logging.basicConfig(level=logging.DEBUG)

app = FastAPI(
    title="Mini Memory Server",
    description="Tiered agent memory with a 5-stage classification pipeline.",
    version="0.1.0",
)

app.include_router(memory_router)


@app.get("/", tags=["health"])
def health() -> dict:
    return {"status": "ok"}
