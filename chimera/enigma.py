"""
enigma.py
=========

Un enigma per CHIMERA: scoprire una FRASE SEGRETA che nessuno le ha detto.

Dimostra due cose, una positiva e una negativa (entrambe oneste):

  1) FRASE con segnale graduale ("quante lettere sono al posto giusto"):
     l'evoluzione la SCOPRE, partendo da stringhe casuali, senza mai vedere
     la risposta. Vince perche' puo' scalare un gradiente: caldo/freddo.

  2) HASH (feedback solo si'/no): il paesaggio e' piatto ovunque tranne un
     punto. L'evoluzione NON fa meglio del caso. E' esattamente il motivo per
     cui gli hash proteggono le password.

Punto chiave d'ingegneria: NON riscriviamo il motore. Riusiamo lo stesso
World/Continent/Concept/Archive del resto di CHIMERA; cambiamo solo il
SUBSTRATO (un genoma-stringa invece di una funzione) e l'AMBIENTE (il segreto).
Le vie numeriche (metamorfosi, apprendimento) restano spente. E' la prova
concreta che "ogni parte del sistema puo' essere sostituita".

Uso:
    python -m chimera.enigma
    python -m chimera.enigma --secret "il senso emerge dagli effetti"
    python -m chimera.enigma --mode hash        # il fallimento istruttivo
"""

from __future__ import annotations

import argparse
import hashlib
import random
import string
import sys
from dataclasses import dataclass

from .core.archive import Archive
from .core.concept import Concept
from .core.evolution import Continent, World

# Alfabeto di ricerca: lettere, spazio e po' di punteggiatura. NON coincide con
# le sole lettere del segreto: l'engine deve cercare in tutto questo spazio.
ALPHABET = string.ascii_letters + " " + ".,!?'"


# --------------------------------------------------------------------------- #
#  SUBSTRATO NUOVO: un genoma-stringa (una sequenza di caratteri che evolve).
#  Rispetta il "contratto" che il motore si aspetta da un substrato:
#  mutate / recombine / describe / complexity / kind.
# --------------------------------------------------------------------------- #
@dataclass
class StringPhysics:
    """La 'fisica' di un continente testuale: alfabeto, lunghezza, tasso di
    mutazione e un RNG proprio. (L'equivalente della Physics numerica.)"""
    alphabet: str
    length: int
    rng: random.Random
    mutation_rate: float = 0.08


@dataclass
class StringGenome:
    chars: list[str]
    kind: str = "string"

    @classmethod
    def random(cls, physics: StringPhysics) -> "StringGenome":
        return cls([physics.rng.choice(physics.alphabet) for _ in range(physics.length)])

    def mutate(self, physics: StringPhysics) -> "StringGenome":
        """Mutazione puntuale: ogni posizione cambia con probabilita' p; se per
        caso nessuna cambia, ne forza una (una mutazione avviene sempre)."""
        new = list(self.chars)
        changed = False
        for i in range(len(new)):
            if physics.rng.random() < physics.mutation_rate:
                new[i] = physics.rng.choice(physics.alphabet)
                changed = True
        if not changed and new:
            i = physics.rng.randrange(len(new))
            new[i] = physics.rng.choice(physics.alphabet)
        return StringGenome(new)

    def recombine(self, other: "StringGenome", physics: StringPhysics) -> "StringGenome":
        """Crossover a un punto: prende un pezzo da un genitore e un pezzo
        dall'altro. E' l'accoppiamento, come tra le formule numeriche."""
        n = len(self.chars)
        if n < 2 or len(other.chars) != n:
            return self.mutate(physics)
        cut = physics.rng.randrange(1, n)
        return StringGenome(list(self.chars[:cut]) + list(other.chars[cut:]))

    def describe(self) -> str:
        return "".join(self.chars)

    def complexity(self) -> int:
        return len(self.chars)


# --------------------------------------------------------------------------- #
#  AMBIENTI: il segreto e il segnale che l'engine riceve (l'unica cosa che vede).
# --------------------------------------------------------------------------- #
class SecretPhrase:
    """Feedback GRADUALE: frazione di lettere al posto giusto. Un gradiente da
    scalare -> l'evoluzione converge. (L'engine riceve solo questo numero.)"""

    def __init__(self, secret: str):
        self._secret = list(secret)
        self.length = len(self._secret)
        self.name = "enigma:frase"

    def evaluate(self, concept: Concept) -> tuple[float, bool]:
        g = concept.substrate.chars
        if len(g) != self.length:
            return float("-inf"), False
        matches = sum(1 for a, b in zip(g, self._secret) if a == b)
        return matches / self.length, matches == self.length

    def sota_score(self) -> float:
        return 1.0


class SecretHash:
    """Feedback PIATTO: 1.0 solo se il genoma ha esattamente lo stesso hash del
    segreto, altrimenti 0.0. Nessun gradiente -> l'evoluzione e' cieca."""

    def __init__(self, secret: str):
        self.length = len(secret)
        self._target = hashlib.sha256(secret.encode()).hexdigest()
        self.name = "enigma:hash"

    def evaluate(self, concept: Concept) -> tuple[float, bool]:
        g = "".join(concept.substrate.chars)
        ok = hashlib.sha256(g.encode()).hexdigest() == self._target
        return (1.0 if ok else 0.0), ok

    def sota_score(self) -> float:
        return 1.0


# --------------------------------------------------------------------------- #
#  Costruzione del mondo (riusa il motore di CHIMERA) e ciclo di scoperta.
# --------------------------------------------------------------------------- #
# Continenti con "fisiche" testuali diverse: tassi di mutazione diversi ->
# esploratori piu' o meno irrequieti. La migrazione mescola i loro ritrovati.
_CONTINENTS = [("Alpha", 0.05), ("Beta", 0.12)]


def _seed_world(env, length: int, pop: int, seed: int) -> World:
    archive = Archive(f"enigma_archive.sqlite", run_id=f"enigma-{seed}")
    archive.clear()
    continents = []
    for i, (name, mrate) in enumerate(_CONTINENTS):
        phys = StringPhysics(ALPHABET, length, random.Random(seed + i + 1), mutation_rate=mrate)
        pop_list = [Concept(substrate=StringGenome.random(phys), continent=name, origin="genesis")
                    for _ in range(pop)]
        continents.append(Continent(name=name, physics=phys, population=pop_list))
    # meta/learn spenti: sono vie numeriche, qui evolvono solo stringhe.
    world = World(env, continents, archive, elitism=0.3, recomb_rate=0.5,
                  meta_rate=0.0, learn_rate=0.0, migration_every=6, migrants=4, seed=seed)
    world.bootstrap()
    return world


def _best(world: World) -> Concept:
    return max((c for cont in world.continents for c in cont.population),
              key=lambda c: c.fitness)


def _mask(guess: str, secret: str) -> str:
    """Rivelazione per l'UMANO (non per l'engine): lettere giuste in chiaro,
    sbagliate come '·'. Serve a VEDERE la frase emergere dal rumore."""
    return "".join(g if g == s else "·" for g, s in zip(guess, secret))


def run(secret: str, mode: str = "frase", generations: int = 600,
        pop: int = 200, seed: int = 0) -> None:
    try:                                     # console Windows: forza UTF-8
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    L = len(secret)
    space = len(ALPHABET) ** L
    env = SecretPhrase(secret) if mode == "frase" else SecretHash(secret)

    print(f"\n  CHIMERA — enigma [{mode}]")
    print(f"  alfabeto: {len(ALPHABET)} simboli · lunghezza: {L} · "
          f"spazio di ricerca: {len(ALPHABET)}^{L} ≈ 10^{len(str(space)) - 1}")
    print(f"  (l'engine NON vede il segreto: riceve solo il punteggio)\n")

    world = _seed_world(env, L, pop, seed)
    best_matches = -1
    solved_at = None

    for _ in range(generations):
        world.step()
        champ = _best(world)
        guess = champ.substrate.describe()
        matches = sum(1 for a, b in zip(guess, secret) if a == b)

        if matches > best_matches:          # stampa solo quando MIGLIORA
            best_matches = matches
            bar = "█" * matches + "░" * (L - matches)
            print(f"  gen {world.generation:4d} | {matches:2d}/{L} {bar} | {_mask(guess, secret)!r}")

        if champ.beats_sota:                 # risolto (hash) o tutte giuste (frase)
            solved_at = world.generation
            break

    tried = world.archive.stats()["total"]
    print()
    if solved_at is not None:
        print(f"  ✔ RISOLTO alla generazione {solved_at}.")
        print(f"    CHIMERA ha scoperto: \"{secret}\"")
        print(f"    Tentativi valutati: {tried:,} su uno spazio di ~10^{len(str(space)) - 1}. "
              f"Non e' forza bruta: e' selezione.")
    else:
        got = _best(world).substrate.describe()
        print(f"  ✗ NON risolto in {generations} generazioni ({tried:,} tentativi).")
        if mode == "hash":
            print(f"    Miglior tentativo: \"{got}\" — {best_matches}/{L} lettere giuste per CASO.")
            print(f"    Il feedback e' solo si'/no: nessun gradiente, nessun progresso possibile.")
            print(f"    E' per questo che un hash protegge una password: l'evoluzione e' cieca qui.")
        else:
            print(f"    Miglior tentativo: \"{got}\" ({best_matches}/{L}). Prova piu' generazioni.")
    world.archive.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="CHIMERA — scoprire una frase segreta per evoluzione")
    ap.add_argument("--secret", default="CHIMERA scopre da sola",
                    help="la frase segreta da riscoprire (l'engine non la vede)")
    ap.add_argument("--mode", default="frase", choices=["frase", "hash"],
                    help="'frase' = feedback graduale (risolvibile); 'hash' = si'/no (cieco)")
    ap.add_argument("--generations", type=int, default=600)
    ap.add_argument("--pop", type=int, default=200, help="concetti per continente")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.secret, args.mode, args.generations, args.pop, args.seed)


if __name__ == "__main__":
    main()
