"""
serve.py
========

Avvia l'osservatorio web di CHIMERA.

Uso:
    python -m chimera.serve
    python -m chimera.serve --pop 150 --delay 0.15 --port 8000

Poi apri il browser su http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse

import uvicorn

from .ui.runner import ENVIRONMENTS, EngineRunner
from .ui.server import create_app


def main() -> None:
    ap = argparse.ArgumentParser(description="CHIMERA v0.2 - osservatorio web dell'evoluzione")
    ap.add_argument("--pop", type=int, default=120, help="concetti per continente")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delay", type=float, default=0.25, help="secondi tra una generazione e l'altra")
    ap.add_argument("--generations", type=int, default=400, help="tetto massimo di generazioni")
    ap.add_argument("--env", default="symbolic_regression",
                    choices=sorted(ENVIRONMENTS.keys()),
                    metavar="AMBIENTE", help="ambiente di prova (incl. feynman:<equazione>)")
    ap.add_argument("--archive", default="chimera_archive.sqlite")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    runner = EngineRunner(
        pop=args.pop, seed=args.seed, delay=args.delay,
        max_generations=args.generations, archive_path=args.archive, env_name=args.env,
    )
    app = create_app(runner)
    print(f"\n  CHIMERA - osservatorio su http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
