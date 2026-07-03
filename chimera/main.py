"""
main.py
=======

Runner da riga di comando per CHIMERA v0.1.

Crea un mondo con piu' continenti (ognuno con la sua matematica), fa evolvere
i concetti su un ambiente concreto (regressione simbolica) e stampa:
    - l'avanzamento generazione per generazione,
    - il miglior concetto trovato,
    - la sua genealogia (l'albero evolutivo ricostruito dall'archivio).

Uso:
    python -m chimera.main --generations 60 --pop 120
"""

from __future__ import annotations

import argparse
import random
import sys

# la console di Windows a volte usa cp1252: forziamo UTF-8 per l'output.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from .core.archive import Archive
from .core.evolution import Continent, World
from .core.substrate import Physics
from .ui.runner import make_environment


def build_continents(pop: int, seed: int) -> list[Continent]:
    """Quattro continenti con FISICHE diverse (matematiche diverse)."""
    rng = lambda s: random.Random(seed + s)

    specs = [
        # nome        operatori unari                         binari
        ("Alpha", ("neg", "abs", "sin", "cos"),               ("add", "sub", "mul")),
        ("Beta",  ("tanh", "square", "neg"),                  ("add", "mul", "div")),
        ("Gamma", ("log", "sqrt", "abs", "square"),           ("add", "sub", "mul", "div")),
        ("Delta", ("sin", "cos", "tanh", "neg", "abs"),       ("add", "sub", "mul", "div")),
    ]
    continents = []
    for i, (name, unary, binary) in enumerate(specs):
        physics = Physics(unary=unary, binary=binary, max_depth=4, rng=rng(i + 1))
        population = World.seed_population(physics, name, pop)
        continents.append(Continent(name=name, physics=physics, population=population))
    return continents


def main() -> None:
    ap = argparse.ArgumentParser(description="CHIMERA v0.1 - laboratorio di evoluzione dei concetti")
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--pop", type=int, default=120, help="concetti per continente")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env", default="symbolic_regression",
                    choices=["symbolic_regression", "compression"], help="ambiente di prova")
    ap.add_argument("--archive", default="chimera_archive.sqlite")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    env = make_environment(args.env, args.seed)
    archive = Archive(args.archive, run_id=f"run-{args.seed}")
    continents = build_continents(args.pop, args.seed)
    world = World(env, continents, archive, seed=args.seed)

    print("=" * 68)
    print(" CHIMERA  -  evoluzione di concetti")
    print("=" * 68)
    print(f" Ambiente          : {env.name}")
    print(f" Stato dell'arte   : fitness baseline = {env.sota_score():.4f}")
    print(f" Continenti        : {', '.join(c.name for c in continents)}")
    print(f" Popolazione/cont. : {args.pop}   |   Generazioni: {args.generations}")
    print("-" * 68)

    world.bootstrap()
    for _ in range(args.generations):
        snap = world.step()
        if not args.quiet:
            line = " | ".join(
                f"{c['continent']}:{c['best_fitness']:+.3f}({c['beats_sota']} SOTA)"
                for c in snap["continents"]
            )
            print(f" gen {snap['generation']:>3}  {line}")

    print("-" * 68)
    best = archive.best(1)[0]
    bid, cont, gen, fit, beats, cx, struct, kind = best
    print(" MIGLIOR CONCETTO")
    print(f"   id            : {bid}")
    print(f"   continente    : {cont}")
    print(f"   forma         : {kind}")
    print(f"   generazione   : {gen}")
    print(f"   fitness       : {fit:.4f}   (baseline {env.sota_score():.4f})")
    print(f"   batte SOTA    : {'SI' if beats else 'no'}")
    print(f"   complessita'  : {cx}")
    print(f"   struttura     : {struct}")

    print("\n GENEALOGIA (dai capostipiti al concetto finale)")
    for row in archive.lineage(bid):
        cid, origin, parents, cfit, cstruct = row
        par = parents if parents else "-"
        print(f"   #{cid:<5} [{origin:<13}] genitori={par:<10} fit={cfit:+.4f}  {cstruct}")

    st = archive.stats()
    print("\n ARCHIVIO EVOLUTIVO")
    print(f"   concetti totali registrati : {st['total']}")
    print(f"   concetti che battono SOTA  : {st['beat_sota']}")
    print(f"   per continente             : {st['per_continent']}")
    print(f"   database                   : {args.archive}")
    archive.close()


if __name__ == "__main__":
    main()
