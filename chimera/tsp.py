"""
tsp.py
======

Il commesso viaggiatore su CITTA' VERE: il giro piu' corto che tocca ogni
citta' una volta sola e torna a casa.

Qui l'enigma e' diverso da tutti gli altri: **nessuno conosce la risposta
ottima** (per N citta' il numero di giri e' (N-1)!/2, impossibile da enumerare).
Non c'e' un "risolto": c'e' solo "il miglior giro trovato finora". E' pura
scoperta / ottimizzazione.

Rappresentazione: una PERMUTAZIONE delle citta' (l'ordine di visita).
Gradiente = lunghezza totale del giro (piu' corto = meglio). Distanze reali
via formula dell'emisenoverso (haversine), in km.

Baseline da battere: l'euristica "vai sempre alla citta' piu' vicina"
(nearest neighbor), la mossa avida piu' ovvia.

Uso:
    python -m chimera.tsp
    python -m chimera.tsp --generations 800 --pop 400
"""

from __future__ import annotations

import argparse
import math
import sys

from .combinatorial import best, seed_world
from .core.concept import Concept

# (nome, latitudine, longitudine) — citta' italiane reali
CITIES = [
    ("Roma", 41.90, 12.50), ("Milano", 45.46, 9.19), ("Napoli", 40.85, 14.27),
    ("Torino", 45.07, 7.69), ("Palermo", 38.12, 13.36), ("Genova", 44.41, 8.93),
    ("Bologna", 44.49, 11.34), ("Firenze", 43.77, 11.26), ("Bari", 41.12, 16.87),
    ("Catania", 37.50, 15.09), ("Venezia", 45.44, 12.32), ("Trieste", 45.65, 13.77),
    ("Cagliari", 39.22, 9.12), ("ReggioCal", 38.11, 15.65), ("Ancona", 43.62, 13.52),
]


def _haversine(a, b) -> float:
    R = 6371.0
    (lat1, lon1), (lat2, lon2) = a, b
    p = math.pi / 180
    dlat, dlon = (lat2 - lat1) * p, (lon2 - lon1) * p
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


class TSP:
    def __init__(self, cities):
        self.cities = cities
        n = len(cities)
        coords = [(c[1], c[2]) for c in cities]
        self.D = [[_haversine(coords[i], coords[j]) for j in range(n)] for i in range(n)]
        self.name = "enigma:commesso"
        self._baseline = self._tour_len(self._nearest_neighbor())

    def _tour_len(self, order: list[int]) -> float:
        n = len(order)
        return sum(self.D[order[i]][order[(i + 1) % n]] for i in range(n))

    def _nearest_neighbor(self) -> list[int]:
        n = len(self.cities)
        unvisited = set(range(1, n))
        order, cur = [0], 0
        while unvisited:
            nxt = min(unvisited, key=lambda j: self.D[cur][j])
            order.append(nxt)
            unvisited.discard(nxt)
            cur = nxt
        return order

    def evaluate(self, concept: Concept) -> tuple[float, bool]:
        length = self._tour_len(concept.substrate.order)
        return -length, length < self._baseline

    def sota_score(self) -> float:
        return -self._baseline

    def optimal_tour(self) -> tuple[list[int], float]:
        """Ottimo esatto via Held-Karp (DP su sottoinsiemi). Trattabile per n<=~18.
        Serve come 'vetta vera' per misurare quanto un paesaggio e' ingannevole."""
        n, D, INF = len(self.cities), self.D, float("inf")
        dp = [[INF] * n for _ in range(1 << n)]
        par = [[-1] * n for _ in range(1 << n)]
        dp[1][0] = 0.0
        for mask in range(1 << n):
            if not mask & 1:
                continue
            for j in range(n):
                if dp[mask][j] == INF:
                    continue
                base = dp[mask][j]
                for k in range(n):
                    if mask & (1 << k):
                        continue
                    nm, c = mask | (1 << k), base + D[j][k]
                    if c < dp[nm][k]:
                        dp[nm][k] = c
                        par[nm][k] = j
        full = (1 << n) - 1
        end = min(range(1, n), key=lambda j: dp[full][j] + D[j][0])
        length = dp[full][end] + D[end][0]
        order, mask, j = [], full, end
        while j != -1:
            order.append(j)
            pj = par[mask][j]
            mask ^= (1 << j)
            j = pj
        order.reverse()
        return order, length


def _rotate_to_zero(order: list[int]) -> list[int]:
    i = order.index(0)
    return order[i:] + order[:i]


def _label(i: int) -> str:
    return chr(ord("A") + i)


def _map(cities, order) -> str:
    W, H = 52, 20
    lats = [c[1] for c in cities]
    lons = [c[2] for c in cities]
    la0, la1, lo0, lo1 = min(lats), max(lats), min(lons), max(lons)
    grid = [[" "] * W for _ in range(H)]
    for idx, (name, lat, lon) in enumerate(cities):
        cx = int((lon - lo0) / (lo1 - lo0 + 1e-9) * (W - 1))
        cy = int((la1 - lat) / (la1 - la0 + 1e-9) * (H - 1))   # nord in alto
        grid[cy][cx] = _label(idx)
    return "\n".join("  " + "".join(row) for row in grid)


def run(generations: int = 700, pop: int = 350, seed: int = 0) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    env = TSP(CITIES)
    n = len(CITIES)
    # (n-1)!/2 giri distinti
    space = math.factorial(n - 1) // 2

    print(f"\n  CHIMERA — enigma: commesso viaggiatore su {n} citta' italiane")
    print(f"  giri possibili: (n-1)!/2 ≈ {space:,}")
    print(f"  baseline (citta' piu' vicina): {env._baseline:,.0f} km")
    print(f"  (nessuno conosce l'ottimo: e' scoperta, non una soluzione nota)\n")

    world = seed_world(env, n, pop, seed, run_id=f"tsp-{seed}")
    best_len = None
    beaten = False

    for _ in range(generations):
        world.step()
        champ = best(world)
        length = env._tour_len(champ.substrate.order)
        if best_len is None or length < best_len - 1.0:
            best_len = length
            gain = 100 * (env._baseline - length) / env._baseline
            print(f"  gen {world.generation:4d} | giro {length:8,.0f} km | "
                  f"vs baseline {gain:+5.1f}%")
        if champ.beats_sota and not beaten:
            beaten = True
            print(f"       ↑ ha superato l'euristica avida alla generazione {world.generation}")

    tried = world.archive.stats()["total"]
    order = _rotate_to_zero(best(world).substrate.order)
    gain = 100 * (env._baseline - best_len) / env._baseline

    print(f"\n  Miglior giro trovato: {best_len:,.0f} km "
          f"({gain:+.1f}% rispetto all'euristica) — {tried:,} tentativi.\n")
    print(_map(CITIES, order))
    legend = "  ".join(f"{_label(i)}={CITIES[i][0]}" for i in range(n))
    print("\n  " + legend)
    route = " → ".join(_label(i) for i in order) + " → " + _label(order[0])
    print("\n  Rotta:  " + route)
    print(f"\n  Nessuno gli ha dato la risposta: l'ha cercata lui tra {space:,} giri possibili.")
    world.archive.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="CHIMERA — commesso viaggiatore per evoluzione")
    ap.add_argument("--generations", type=int, default=700)
    ap.add_argument("--pop", type=int, default=350)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.generations, args.pop, args.seed)


if __name__ == "__main__":
    main()
