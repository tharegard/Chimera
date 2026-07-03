"""
queens.py
=========

Enigma delle N regine: disporre N regine su una scacchiera N×N in modo che
nessuna ne minacci un'altra (righe, colonne, diagonali).

Rappresentazione: una PERMUTAZIONE di lunghezza N. `order[c] = r` significa
"nella colonna c la regina sta sulla riga r". Una permutazione garantisce
GIA' che non ci siano due regine sulla stessa riga o colonna: all'evoluzione
resta solo da sistemare le DIAGONALI. Gradiente = numero di coppie che si
attaccano in diagonale (0 = risolto).

Uso:
    python -m chimera.queens
    python -m chimera.queens --n 12 --generations 400
"""

from __future__ import annotations

import argparse
import sys

from .combinatorial import best, seed_world
from .core.concept import Concept


class NQueens:
    def __init__(self, n: int = 8):
        self.n = n
        self.name = f"enigma:{n}-regine"

    def _conflicts(self, order: list[int]) -> int:
        n = len(order)
        c = 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(order[i] - order[j]) == abs(i - j):   # stessa diagonale
                    c += 1
        return c

    def evaluate(self, concept: Concept) -> tuple[float, bool]:
        c = self._conflicts(concept.substrate.order)
        return -float(c), c == 0

    def sota_score(self) -> float:
        return 0.0


def _board(order: list[int]) -> str:
    n = len(order)
    out = []
    for r in range(n):
        row = " ".join("Q" if order[c] == r else "." for c in range(n))
        out.append("  " + row)
    return "\n".join(out)


def run(n: int = 8, generations: int = 400, pop: int = 200, seed: int = 0) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    env = NQueens(n)
    total_pairs = n * (n - 1) // 2
    print(f"\n  CHIMERA — enigma: {n} regine su scacchiera {n}×{n}")
    print(f"  coppie di regine da rendere pacifiche: {total_pairs} · "
          f"disposizioni possibili: {n}! ")
    print(f"  (l'engine NON conosce una soluzione: riceve solo il n. di conflitti)\n")

    world = seed_world(env, n, pop, seed, run_id=f"queens-{seed}")
    best_conf = None
    solved_at = None

    for _ in range(generations):
        world.step()
        champ = best(world)
        conf = env._conflicts(champ.substrate.order)
        if best_conf is None or conf < best_conf:
            best_conf = conf
            print(f"  gen {world.generation:4d} | conflitti {conf:3d} | "
                  f"coppie pacifiche {total_pairs - conf}/{total_pairs}")
        if champ.beats_sota:
            solved_at = world.generation
            break

    tried = world.archive.stats()["total"]
    print()
    if solved_at is not None:
        print(f"  ✔ RISOLTO alla generazione {solved_at} ({tried:,} tentativi):\n")
        print(_board(best(world).substrate.order))
        print(f"\n    Nessuna regina minaccia le altre. Scoperto, non calcolato a tavolino.")
    else:
        print(f"  ✗ Non risolto in {generations} generazioni: miglior tentativo {best_conf} conflitti.")
    world.archive.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="CHIMERA — N regine per evoluzione")
    ap.add_argument("--n", type=int, default=8, help="numero di regine / lato scacchiera")
    ap.add_argument("--generations", type=int, default=400)
    ap.add_argument("--pop", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.n, args.generations, args.pop, args.seed)


if __name__ == "__main__":
    main()
