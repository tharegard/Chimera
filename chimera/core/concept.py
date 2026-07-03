"""
concept.py
==========

Un Concetto e' l'unita' di evoluzione di CHIMERA.

Non e' una parola, non ha significato umano. E' un oggetto con:
    - un substrato (la sua forma concreta: oggi un ExpressionTree)
    - una memoria evolutiva (genitori, mutazione che l'ha generato, fitness)
    - un'identita' stabile (id) per ricostruire l'albero genealogico.

Il significato di un concetto non e' scritto qui: emerge dalla fitness che
l'ambiente gli assegna in base agli effetti che produce.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .substrate import Substrate

_id_counter = itertools.count(1)


@dataclass
class Concept:
    substrate: Substrate
    continent: str = ""
    generation: int = 0
    parents: tuple[int, ...] = ()
    origin: str = "genesis"          # "genesis" | "mutation" | "recombination" | "migration"
    fitness: float = float("-inf")   # assegnata dall'ambiente; piu' alta = migliore
    beats_sota: bool = False         # ha battuto lo stato dell'arte?
    id: int = field(default_factory=lambda: next(_id_counter))
    children_ids: list[int] = field(default_factory=list)

    # -- riproduzione ------------------------------------------------------ #
    def mutated(self, physics) -> "Concept":
        return Concept(
            substrate=self.substrate.mutate(physics),
            continent=self.continent,
            generation=self.generation + 1,
            parents=(self.id,),
            origin="mutation",
        )

    def recombined_with(self, other: "Concept", physics) -> "Concept":
        return Concept(
            substrate=self.substrate.recombine(other.substrate, physics),
            continent=self.continent,
            generation=max(self.generation, other.generation) + 1,
            parents=(self.id, other.id),
            origin="recombination",
        )

    def metamorphosed(self, physics) -> "Concept":
        """Il concetto cambia FORMA preservando il comportamento (v0.3).
        Se la metamorfosi non e' possibile, ricade su una normale mutazione.
        """
        from .substrate import metamorphose
        new_sub = metamorphose(self.substrate, physics)
        if new_sub is None:
            return self.mutated(physics)
        return Concept(
            substrate=new_sub,
            continent=self.continent,
            generation=self.generation + 1,
            parents=(self.id,),
            origin="metamorphosis",
        )

    def learned(self, x, y, physics) -> "Concept":
        """APPRENDIMENTO entro la vita: adatta i pesi all'ambiente (minimi quadrati).
        Se la forma attuale non sa apprendere in forma chiusa (un programma), prima
        si TRASFORMA in una forma apprendibile (vettore/rete), poi impara."""
        from .substrate import metamorphose
        new_sub = self.substrate.learn(x, y)
        if new_sub is self.substrate:                 # forma non apprendibile
            morph = metamorphose(self.substrate, physics)
            if morph is None:
                return self.mutated(physics)
            new_sub = morph.learn(x, y)
        return Concept(
            substrate=new_sub,
            continent=self.continent,
            generation=self.generation + 1,
            parents=(self.id,),
            origin="learning",
        )

    @property
    def substrate_kind(self) -> str:
        return self.substrate.kind

    def migrated_to(self, continent: str) -> "Concept":
        # la migrazione non copia: lo stesso individuo cambia continente,
        # mantenendo la sua genealogia intatta.
        self.continent = continent
        self.origin = "migration"
        return self

    def describe(self) -> str:
        return self.substrate.describe()
