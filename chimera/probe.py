"""
probe.py
========

CHIMERA come STRUMENTO DI MISURA, non come solutore.

Rovesciamo la domanda: non "il motore risolve il problema?" ma "che forma ha il
paesaggio del problema?". Perché è la forma del terreno, non la potenza del
motore, a decidere se un problema è facile o difficile.

Metafora: fitness = altitudine, risolvere = arrivare in cima, il motore = un
escursionista nella nebbia che sente solo se un passo va in salita. Misuriamo:

  1. ACCIDENTATO?  (ruggedness) — cammino a caso e guardo quanto in fretta
     l'altitudine "dimentica" da dove veniva. λ = lunghezza di correlazione.
     La riporto NORMALIZZATA: λ/diametro-dello-spazio, così problemi di taglia
     diversa sono confrontabili (un passo grande vale meno su spazi grandi).
  2. SALI SPESSO?  (densità di gradiente) — da punti a caso, quale frazione dei
     passi va in salita.
  3. PIATTO?       (neutralità) — frazione di passi a pari altitudine (plateau).
  4. TRAPPOLE?     (tasso di ottimi locali) — lancio salite in gradiente da tanti
     punti casuali; quante restano incastrate in una cima che NON è la soluzione.
     Non serve conoscere gli ottimi -> onesto anche con molte soluzioni, e
     confrontabile tra encoding diversi (è una frazione).
  5. RISOLVE / IN FRETTA — dinamica reale del motore: su più semi, quante volte
     arriva in cima e in quante generazioni.

ONESTA': misura la difficoltà PER QUESTA DINAMICA FISSA (questo motore, questi
operatori). È un'impronta del landscape, non una verità assoluta: cambiando
l'encoding, il terreno cambia. Tra encoding diversi restano confrontabili solo
le metriche adimensionali (salita, piatto, trappole, e λ/diametro con cautela).

Uso:
    python -m chimera.probe
"""

from __future__ import annotations

import math
import random
import statistics
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Iterator

from .combinatorial import Permutation, best, seed_world
from .cryptarithm import Cryptarithm
from .enigma import ALPHABET, SecretHash, SecretPhrase
from .enigma import _seed_world as string_seed_world
from .queens import NQueens
from .tsp import CITIES, TSP

EPS = 1e-9


# --------------------------------------------------------------------------- #
#  Un "bersaglio" della sonda: tutto ciò che serve per misurare un paesaggio,
#  indipendente dall'encoding (permutazione o stringa).
# --------------------------------------------------------------------------- #
@dataclass
class Target:
    name: str
    encoding: str                       # "perm" | "str"
    diameter: int                       # distanza massima nello spazio
    cap: int                            # generazioni max per il solver
    traps: int                          # quanti hill-climb per il tasso di trappole
    rand: Callable[[random.Random], list]
    neighbor: Callable[[list, random.Random], list]
    neighbors_iter: Callable[[list, random.Random], Iterator[list]]
    fit: Callable[[list], float]
    is_global: Callable[[list], bool]
    solve: Callable[[int], tuple]       # seed -> (gen_or_None, genoma)


# --------------------------------------------------------------------------- #
#  Metriche (un'unica sorgente, valgono per ogni encoding).
# --------------------------------------------------------------------------- #
def ruggedness(t: Target, steps: int = 1500, seed: int = 0) -> float:
    rng = random.Random(seed)
    g = t.rand(rng)
    series = []
    for _ in range(steps):
        series.append(t.fit(g))
        g = t.neighbor(g, rng)
    m = statistics.fmean(series)
    denom = sum((s - m) ** 2 for s in series)
    if denom == 0:
        return float("nan")            # paesaggio costante: ruggedness indefinita
    r1 = sum((series[k] - m) * (series[k + 1] - m) for k in range(len(series) - 1)) / denom
    lam = (-1.0 / math.log(r1)) if 0 < r1 < 1 else (float("inf") if r1 >= 1 else 0.0)
    return lam / t.diameter if lam != float("inf") else float("inf")


def gradient_stats(t: Target, samples: int = 100, neighbors: int = 20, seed: int = 1):
    rng = random.Random(seed)
    up = flat = total = 0
    for _ in range(samples):
        g = t.rand(rng)
        f = t.fit(g)
        for _ in range(neighbors):
            f2 = t.fit(t.neighbor(g, rng))
            total += 1
            if f2 > f + EPS:
                up += 1
            elif abs(f2 - f) <= EPS:
                flat += 1
    return (up / total, flat / total) if total else (0.0, 0.0)


def trap_rate(t: Target, seed: int = 2) -> float:
    """Da tanti punti casuali, sali sempre (first-improvement) fino a bloccarti
    in una cima locale. Quante di queste cime NON sono la soluzione = trappole."""
    rng = random.Random(seed)
    stuck_bad = 0
    for _ in range(t.traps):
        g = t.rand(rng)
        f = t.fit(g)
        while True:
            improved = False
            for nb in t.neighbors_iter(g, rng):     # ordine casuale
                f2 = t.fit(nb)
                if f2 > f + EPS:
                    g, f, improved = nb, f2, True
                    break
            if not improved:                        # cima locale raggiunta
                break
        if not t.is_global(g):
            stuck_bad += 1
    return stuck_bad / t.traps


def measure(t: Target, ga_seeds=range(4)) -> dict:
    lam = ruggedness(t)
    p_up, p_flat = gradient_stats(t)
    trap = trap_rate(t)
    gens = []
    for s in ga_seeds:
        g, _ = t.solve(s)
        if g is not None:
            gens.append(g)
    n_seeds = len(list(ga_seeds))
    return {
        "name": t.name, "enc": t.encoding, "lam": lam, "p_up": p_up,
        "p_flat": p_flat, "trap": trap, "success": len(gens) / n_seeds,
        "med_gen": statistics.median(gens) if gens else None,
    }


# --------------------------------------------------------------------------- #
#  Costruttori di bersagli.
# --------------------------------------------------------------------------- #
def _perm_target(name, env, n, is_global, cap, traps, pop) -> Target:
    def rand(rng):
        o = list(range(n))
        rng.shuffle(o)
        return o

    def neighbor(g, rng):
        h = list(g)
        i, j = rng.randrange(n), rng.randrange(n)
        h[i], h[j] = h[j], h[i]
        return h

    def neighbors_iter(g, rng):
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(pairs)
        for i, j in pairs:
            h = list(g)
            h[i], h[j] = h[j], h[i]
            yield h

    def fit(g):
        return env.evaluate(SimpleNamespace(substrate=SimpleNamespace(order=g)))[0]

    def solve(seed):
        world = seed_world(env, n, pop, seed, run_id=f"probe-{name}-{seed}")
        gen = None
        for gg in range(1, cap + 1):
            world.step()
            if is_global(best(world).substrate.order):
                gen = gg
                break
        order = list(best(world).substrate.order)
        world.archive.close()
        return gen, order

    return Target(name, "perm", n - 1, cap, traps, rand, neighbor,
                  neighbors_iter, fit, is_global, solve)


def _string_target(name, env, length, alphabet, cap, traps, pop) -> Target:
    def rand(rng):
        return [rng.choice(alphabet) for _ in range(length)]

    def neighbor(g, rng):
        h = list(g)
        i = rng.randrange(length)
        c = rng.choice(alphabet)
        while c == h[i]:
            c = rng.choice(alphabet)
        h[i] = c
        return h

    def neighbors_iter(g, rng):
        moves = [(i, c) for i in range(length) for c in alphabet if c != g[i]]
        rng.shuffle(moves)
        for i, c in moves:
            h = list(g)
            h[i] = c
            yield h

    def fit(g):
        return env.evaluate(SimpleNamespace(substrate=SimpleNamespace(chars=g)))[0]

    def is_global(g):
        return env.evaluate(SimpleNamespace(substrate=SimpleNamespace(chars=g)))[1]

    def solve(seed):
        world = string_seed_world(env, length, pop, seed)
        gen = None
        for gg in range(1, cap + 1):
            world.step()
            if env.evaluate(best(world))[1]:
                gen = gg
                break
        chars = list(best(world).substrate.chars)
        world.archive.close()
        return gen, chars

    return Target(name, "str", length, cap, traps, rand, neighbor,
                  neighbors_iter, fit, is_global, solve)


class _OrderAsPerm:
    """Adatta un ambiente che legge substrate.perm (criptaritmo) al genoma
    condiviso che espone substrate.order."""
    def __init__(self, env):
        self._env = env
        self.name = env.name

    def evaluate(self, concept):
        return self._env.evaluate(SimpleNamespace(substrate=SimpleNamespace(perm=concept.substrate.order)))

    def sota_score(self):
        return self._env.sota_score()


def build_targets() -> list[Target]:
    targets = []

    # Regine (permutazione): globale = zero conflitti
    for n, cap, traps in [(8, 60, 30), (30, 200, 15)]:
        q = NQueens(n)
        targets.append(_perm_target(f"Regine (n={n})", q, n,
                                    lambda o, q=q: q._conflicts(o) == 0, cap, traps, 200))

    # Commesso (permutazione): globale = a <=1% dall'ottimo esatto (Held-Karp)
    tsp = TSP(CITIES)
    _, opt_len = tsp.optimal_tour()
    tsp_n = len(CITIES)
    targets.append(_perm_target(f"Commesso (n={tsp_n})", tsp, tsp_n,
                                lambda o: tsp._tour_len(o) <= opt_len * 1.01, 140, 30, 300))

    # Criptaritmo (permutazione di 10 cifre): globale = somma esatta
    crypt = _OrderAsPerm(Cryptarithm(["SEND", "MORE"], "MONEY"))
    targets.append(_perm_target("Criptaritmo (n=10)", crypt, 10,
                                lambda o: crypt.evaluate(SimpleNamespace(substrate=SimpleNamespace(order=o)))[1],
                                200, 30, 200))

    # Frase segreta (stringa): feedback graduale -> globale = tutte le lettere giuste
    secret = "chimera vive"
    targets.append(_string_target(f"Frase (L={len(secret)})", SecretPhrase(secret),
                                  len(secret), ALPHABET, 400, 25, 200))

    # Stessa frase dietro un HASH (stringa): feedback piatto sì/no -> il caso impossibile
    targets.append(_string_target(f"Hash (L={len(secret)})", SecretHash(secret),
                                  len(secret), ALPHABET, 80, 25, 200))

    return targets


def _fmt(r: dict) -> str:
    lam = "n/d" if r["lam"] != r["lam"] else ("∞" if r["lam"] == float("inf") else f"{r['lam']:.2f}")
    med = "—" if r["med_gen"] is None else f"{r['med_gen']:.0f}"
    return (f"  {r['name']:<18} {r['enc']:<4} | {lam:>5} | {r['p_up']*100:5.1f}% | "
            f"{r['p_flat']*100:5.1f}% | {r['trap']*100:5.1f}% | {r['success']*100:4.0f}% | {med:>4}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("\n  CHIMERA — sonda della struttura dei problemi\n")
    print("  problema           enc  |  λ/D  | salita | piatto | trappole | risolve | gen")
    print("  " + "-" * 74)
    for t in build_targets():
        print(_fmt(measure(t)))

    print("\n  Come leggerla:")
    print("   • λ/D (accidentato): grande = liscio · piccolo = frastagliato (normalizzato sul diametro)")
    print("   • salita:  quanto spesso un passo va in su      • piatto: plateau senza pendenza")
    print("   • trappole: % di salite che si bloccano su una cima che NON è la soluzione")
    print("   • risolve/gen: quante volte il motore arriva in cima, in quante generazioni")
    print("\n  enc = encoding. Tra encoding diversi (perm vs str) confronta salita/piatto/trappole")
    print("  (adimensionali); λ/D solo con cautela. È l'impronta del paesaggio per QUESTO motore.\n")


if __name__ == "__main__":
    main()
