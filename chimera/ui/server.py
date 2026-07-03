"""
server.py
=========

API locale di visualizzazione per CHIMERA (FastAPI).

Il browser interroga questi endpoint solo per OSSERVARE l'evoluzione:
    GET  /                  -> la dashboard (index.html)
    GET  /api/state         -> stato corrente (generazione, continenti, campione...)
    GET  /api/lineage/{id}  -> albero genealogico di un concetto
    POST /api/control       -> {action: start|pause|reset|speed, value?: float}

L'evoluzione NON avviene qui: gira nel thread di background di EngineRunner.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from .runner import EngineRunner

_HTML = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
_TREE_HTML = (Path(__file__).parent / "tree.html").read_text(encoding="utf-8")


def create_app(runner: EngineRunner | None = None) -> FastAPI:
    app = FastAPI(title="CHIMERA - Osservatorio", version="0.2.0")
    app.state.runner = runner or EngineRunner()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _HTML

    @app.get("/albero", response_class=HTMLResponse)
    def tree_page():
        return _TREE_HTML

    @app.get("/api/state")
    def state():
        return JSONResponse(app.state.runner.get_state())

    @app.get("/api/tree")
    def tree(since_nodes: int = 0, since_edges: int = 0):
        return JSONResponse(app.state.runner.get_tree(since_nodes, since_edges))

    @app.get("/api/lineage/{concept_id}")
    def lineage(concept_id: int):
        return JSONResponse(app.state.runner.get_lineage(concept_id))

    @app.get("/api/translate")
    def translate():
        return JSONResponse(app.state.runner.get_translation())

    @app.post("/api/control")
    async def control(payload: dict):
        r = app.state.runner
        action = payload.get("action")
        if action == "start":
            r.start()
        elif action == "pause":
            r.pause()
        elif action == "reset":
            r.reset()
        elif action == "speed":
            r.set_speed(payload.get("value", 0.25))
        return {"ok": True, "running": r.get_state().get("running", False)}

    return app
