"""
combinatorial.py
================

Substrato condiviso per gli enigmi combinatori: una PERMUTAZIONE di lunghezza n.

Molti enigmi (8 regine, commesso viaggiatore, assegnazioni) hanno la stessa
struttura: un ORDINE senza ripetizioni. Una permutazione lo cattura per
costruzione, e le sue operazioni evolutive sono standard:
    - mutazione     = scambio di due posizioni (swap)
    - accoppiamento = order crossover (OX): un segmento da un genitore, il
                      resto nell'ordine dell'altro -> resta una permutazione

Come sempre in CHIMERA: NON riscriviamo il motore. Questo e' solo un nuovo
substrato; World/Continent/Concept/Archive restano quelli di sempre.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .core.archive import Archive
from .core.concept import Concept
from .core.evolution import Continent, World


@dataclass
class PermPhysics:
    rng: random.Random
    n: int
    swaps: int = 1            # coppie scambiate a ogni mutazione (intensita')


@dataclass
class Permutation:
    order: list[int]
    kind: str = "perm"

    @classmethod
    def random(cls, physics: PermPhysics) -> "Permutation":
        p = list(range(physics.n))
        physics.rng.shuffle(p)
        return cls(p)

    def mutate(self, physics: PermPhysics) -> "Permutation":
        p = list(self.order)
        n = len(p)
        for _ in range(max(1, physics.swaps)):
            i, j = physics.rng.randrange(n), physics.rng.randrange(n)
            p[i], p[j] = p[j], p[i]
        return Permutation(p)

    def recombine(self, other: "Permutation", physics: PermPhysics) -> "Permutation":
        n = len(self.order)
        if n < 2 or len(other.order) != n:
            return self.mutate(physics)
        a, b = sorted(physics.rng.sample(range(n), 2))
        child: list[int | None] = [None] * n
        child[a:b] = self.order[a:b]
        present = set(self.order[a:b])
        fill = [x for x in other.order if x not in present]
        k = 0
        for i in range(n):
            if child[i] is None:
                child[i] = fill[k]
                k += 1
        return Permutation(child)  # type: ignore[arg-type]

    def describe(self) -> str:
        return ",".join(str(i) for i in self.order)

    def complexity(self) -> int:
        return len(self.order)


# Due continenti con intensita' di mutazione diversa: esploratori piu' o meno
# irrequieti, mescolati dalla migrazione.
DEFAULT_CONTINENTS = [("Alpha", 1), ("Beta", 3)]


def seed_world(env, n: int, pop: int, seed: int, *, run_id: str,
               continents=DEFAULT_CONTINENTS, elitism: float = 0.3,
               recomb_rate: float = 0.6, migration_every: int = 6,
               migrants: int = 4) -> World:
    archive = Archive("enigma_archive.sqlite", run_id=run_id)
    archive.clear()
    conts = []
    for i, (name, swaps) in enumerate(continents):
        phys = PermPhysics(random.Random(seed + i + 1), n, swaps=swaps)
        pop_list = [Concept(substrate=Permutation.random(phys), continent=name, origin="genesis")
                    for _ in range(pop)]
        conts.append(Continent(name=name, physics=phys, population=pop_list))
    world = World(env, conts, archive, elitism=elitism, recomb_rate=recomb_rate,
                  meta_rate=0.0, learn_rate=0.0, migration_every=migration_every,
                  migrants=migrants, seed=seed)
    world.bootstrap()
    return world


def best(world: World) -> Concept:
    return max((c for cont in world.continents for c in cont.population),
              key=lambda c: c.fitness)
