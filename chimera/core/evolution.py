"""
evolution.py
============

Il motore evolutivo di CHIMERA.

Un Continente e' una popolazione con la sua PROPRIA fisica (operatori/costanti
ammessi): continenti diversi esplorano matematiche diverse. Ogni tanto i
migliori concetti MIGRANO da un continente all'altro: e' esattamente cio' che
accelera l'evoluzione quando ecosistemi diversi si incontrano.

Ciclo per generazione:
    valuta -> seleziona -> (mutazione + accoppiamento) -> nuova generazione
    ... e ogni N generazioni: migrazione tra continenti.

Tutto viene registrato nell'Archivio: nulla di cio' che nasce va perduto.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from .archive import Archive
from .concept import Concept
from .substrate import ExpressionTree, Physics


@dataclass
class Continent:
    name: str
    physics: Physics
    population: list[Concept]


class World:
    def __init__(
        self,
        environment,
        continents: list[Continent],
        archive: Archive,
        elitism: float = 0.3,
        recomb_rate: float = 0.35,
        meta_rate: float = 0.15,
        learn_rate: float = 0.2,
        migration_every: int = 8,
        migrants: int = 3,
        seed: int = 0,
    ):
        self.env = environment
        self.continents = continents
        self.archive = archive
        self.elitism = elitism
        self.recomb_rate = recomb_rate
        self.meta_rate = meta_rate           # probabilita' di metamorfosi (v0.3)
        self.learn_rate = learn_rate         # probabilita' di apprendimento (v0.4)
        self.migration_every = migration_every
        self.migrants = migrants
        self.rng = random.Random(seed)
        self.generation = 0
        self.last_births: Counter = Counter()
        # bersaglio di apprendimento (x, y) fornito dall'ambiente
        self._learn_target = environment.learn_target() if hasattr(environment, "learn_target") else None

        # v0.3: ogni continente deve poter distillare comportamenti sugli
        # input dell'ambiente quando un concetto cambia forma.
        probe = getattr(environment, "x", None)
        if probe is not None:
            for cont in continents:
                cont.physics.attach_probe(probe)

    # -- fabbrica di popolazioni iniziali (genesi) ------------------------- #
    @staticmethod
    def seed_population(physics: Physics, continent: str, size: int) -> list[Concept]:
        pop = []
        for _ in range(size):
            c = Concept(
                substrate=ExpressionTree.random(physics),
                continent=continent,
                origin="genesis",
            )
            pop.append(c)
        return pop

    # -- valutazione ------------------------------------------------------- #
    def _score(self, concepts: list[Concept]) -> None:
        for c in concepts:
            if c.fitness == float("-inf"):
                c.fitness, c.beats_sota = self.env.evaluate(c)

    # -- selezione + riproduzione in un continente ------------------------- #
    def _next_generation(self, cont: Continent) -> list[Concept]:
        pop = cont.population
        pop.sort(key=lambda c: c.fitness, reverse=True)
        n = len(pop)
        n_elite = max(1, int(self.elitism * n))
        survivors = pop[:n_elite]

        offspring: list[Concept] = []
        while len(survivors) + len(offspring) < n:
            parent = self._tournament(survivors)
            roll = self.rng.random()
            if roll < self.recomb_rate and len(survivors) > 1:
                mate = self._tournament(survivors)
                child = parent.recombined_with(mate, cont.physics)
            elif roll < self.recomb_rate + self.meta_rate:
                child = parent.metamorphosed(cont.physics)   # cambio di forma
            elif roll < self.recomb_rate + self.meta_rate + self.learn_rate and self._learn_target is not None:
                child = parent.learned(*self._learn_target, cont.physics)  # apprendimento
            else:
                child = parent.mutated(cont.physics)
            offspring.append(child)
            self.last_births[child.origin] += 1

        # registra i genitori nella genealogia dei figli
        new_pop = survivors + offspring
        self._score(offspring)
        self.archive.record_many(offspring)
        return new_pop

    def _tournament(self, pool: list[Concept], k: int = 3) -> Concept:
        contenders = self.rng.sample(pool, min(k, len(pool)))
        return max(contenders, key=lambda c: c.fitness)

    # -- migrazione tra continenti ---------------------------------------- #
    def _migrate(self) -> None:
        if len(self.continents) < 2:
            return
        for i, cont in enumerate(self.continents):
            dest = self.continents[(i + 1) % len(self.continents)]
            best = sorted(cont.population, key=lambda c: c.fitness, reverse=True)[: self.migrants]
            for c in best:
                # un CLONE strutturale migra (l'originale resta a casa)
                traveller = Concept(
                    substrate=c.substrate,
                    continent=dest.name,
                    generation=c.generation,
                    parents=(c.id,),
                    origin="migration",
                )
                traveller.fitness, traveller.beats_sota = self.env.evaluate(traveller)
                dest.population.append(traveller)
                self.archive.record(traveller)
        self.archive.conn.commit()

    # -- ciclo principale -------------------------------------------------- #
    def bootstrap(self) -> None:
        for cont in self.continents:
            self._score(cont.population)
            self.archive.record_many(cont.population)

    def step(self) -> dict:
        self.generation += 1
        self.last_births = Counter()
        for cont in self.continents:
            cont.population = self._next_generation(cont)
        if self.generation % self.migration_every == 0:
            self._migrate()
        return self.snapshot()

    def snapshot(self) -> dict:
        rows = []
        for cont in self.continents:
            best = max(cont.population, key=lambda c: c.fitness)
            winners = sum(1 for c in cont.population if c.beats_sota)
            rows.append(
                {
                    "continent": cont.name,
                    "size": len(cont.population),
                    "best_fitness": best.fitness,
                    "best_id": best.id,
                    "beats_sota": winners,
                }
            )
        return {"generation": self.generation, "continents": rows}
