"""
cryptarithm.py
==============

Un enigma VERO per CHIMERA: SEND + MORE = MONEY.

Un criptaritmo: ogni lettera vale una cifra diversa (0-9), e la somma deve
tornare esatta. La risposta non la scegliamo noi — e' UNICA e imposta
dall'aritmetica. CHIMERA la deve scoprire dalle regole.

    S E N D            9 5 6 7
  + M O R E    -->   + 1 0 8 5
  ---------          ---------
  M O N E Y          1 0 6 5 2      (soluzione unica)

Rappresentazione (la chiave del perche' funziona):
  un genoma = una PERMUTAZIONE delle cifre 0..9. Le lettere distinte prendono
  le prime posizioni. Una permutazione non ha ripetizioni -> il vincolo
  "cifre tutte diverse" e' rispettato per costruzione, senza sprecare
  l'evoluzione a scartare assegnazioni illegali.
    - mutazione   = scambio di due cifre (swap)
    - accoppiamento = order crossover (OX): preserva la permutazione
    - gradiente   = |SEND + MORE - MONEY|  (0 = risolto)

Come sempre: NON riscriviamo il motore. Riusiamo World/Continent/Concept/
Archive; cambiano solo il substrato (permutazione) e l'ambiente (l'equazione).

Uso:
    python -m chimera.cryptarithm
    python -m chimera.cryptarithm --addends SEND MORE --result MONEY
    python -m chimera.cryptarithm --addends TWO TWO --result FOUR
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass

from .core.archive import Archive
from .core.concept import Concept
from .core.evolution import Continent, World


# --------------------------------------------------------------------------- #
#  SUBSTRATO: un genoma-permutazione (assegna cifre distinte alle lettere).
# --------------------------------------------------------------------------- #
@dataclass
class PermPhysics:
    rng: random.Random
    swaps: int = 1            # quante coppie scambiare a ogni mutazione


@dataclass
class PermGenome:
    perm: list[int]           # una permutazione di 0..9
    kind: str = "perm"

    @classmethod
    def random(cls, physics: PermPhysics) -> "PermGenome":
        p = list(range(10))
        physics.rng.shuffle(p)
        return cls(p)

    def mutate(self, physics: PermPhysics) -> "PermGenome":
        p = list(self.perm)
        for _ in range(max(1, physics.swaps)):
            i, j = physics.rng.randrange(10), physics.rng.randrange(10)
            p[i], p[j] = p[j], p[i]
        return PermGenome(p)

    def recombine(self, other: "PermGenome", physics: PermPhysics) -> "PermGenome":
        """Order crossover (OX): prende un segmento da un genitore e completa
        con le cifre mancanti nell'ordine dell'altro. Resta una permutazione."""
        n = 10
        a, b = sorted(physics.rng.sample(range(n), 2))
        child: list[int | None] = [None] * n
        child[a:b] = self.perm[a:b]
        present = set(child[a:b])
        fill = [x for x in other.perm if x not in present]
        k = 0
        for i in range(n):
            if child[i] is None:
                child[i] = fill[k]
                k += 1
        return PermGenome(child)  # type: ignore[arg-type]

    def describe(self) -> str:
        return "".join(str(d) for d in self.perm)

    def complexity(self) -> int:
        return 10


# --------------------------------------------------------------------------- #
#  AMBIENTE: l'equazione. L'unica cosa che l'engine vede e' l'errore.
# --------------------------------------------------------------------------- #
class Cryptarithm:
    def __init__(self, addends: list[str], result: str):
        self.addends = [w.upper() for w in addends]
        self.result = result.upper()
        words = self.addends + [self.result]
        self.letters = sorted(set("".join(words)))
        if len(self.letters) > 10:
            raise ValueError(f"troppe lettere distinte ({len(self.letters)}): max 10")
        self.leading = {w[0] for w in words}       # una cifra iniziale non e' 0
        self.name = "enigma:criptaritmo"

        # peso posizionale con segno: SEND + MORE - MONEY = somma_L coef[L]*cifra[L]
        self.coef = {L: 0 for L in self.letters}
        for w in self.addends:
            for i, ch in enumerate(reversed(w)):
                self.coef[ch] += 10 ** i
        for i, ch in enumerate(reversed(self.result)):
            self.coef[ch] -= 10 ** i

    def digits_of(self, genome: PermGenome) -> dict[str, int]:
        return {L: genome.perm[i] for i, L in enumerate(self.letters)}

    def evaluate(self, concept: Concept) -> tuple[float, bool]:
        dig = self.digits_of(concept.substrate)
        total = sum(self.coef[L] * dig[L] for L in self.letters)
        err = abs(total)
        lead_bad = any(dig[L] == 0 for L in self.leading)
        fitness = -float(err) - (1e7 if lead_bad else 0.0)   # penalita' forte
        solved = err == 0 and not lead_bad
        return fitness, solved

    def sota_score(self) -> float:
        return 0.0


# --------------------------------------------------------------------------- #
#  Costruzione del mondo (motore riusato) e ciclo di scoperta.
# --------------------------------------------------------------------------- #
_CONTINENTS = [("Alpha", 1), ("Beta", 2)]     # intensita' di mutazione diverse


def _seed_world(env: Cryptarithm, pop: int, seed: int) -> World:
    archive = Archive("enigma_archive.sqlite", run_id=f"crypt-{seed}")
    archive.clear()
    continents = []
    for i, (name, swaps) in enumerate(_CONTINENTS):
        phys = PermPhysics(random.Random(seed + i + 1), swaps=swaps)
        pop_list = [Concept(substrate=PermGenome.random(phys), continent=name, origin="genesis")
                    for _ in range(pop)]
        continents.append(Continent(name=name, physics=phys, population=pop_list))
    world = World(env, continents, archive, elitism=0.3, recomb_rate=0.6,
                  meta_rate=0.0, learn_rate=0.0, migration_every=6, migrants=4, seed=seed)
    world.bootstrap()
    return world


def _best(world: World) -> Concept:
    return max((c for cont in world.continents for c in cont.population),
              key=lambda c: c.fitness)


def _word_value(word: str, dig: dict[str, int]) -> int:
    v = 0
    for ch in word:
        v = v * 10 + dig[ch]
    return v


def _render(env: Cryptarithm, dig: dict[str, int]) -> str:
    a = [str(_word_value(w, dig)) for w in env.addends]
    r = _word_value(env.result, dig)
    return " + ".join(a) + f" = {r}"


def run(addends: list[str], result: str, generations: int = 500,
        pop: int = 300, seed: int = 0) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    env = Cryptarithm(addends, result)
    # spazio: assegnazioni di cifre distinte alle lettere = 10 P k
    k = len(env.letters)
    space = 1
    for i in range(k):
        space *= (10 - i)

    print(f"\n  CHIMERA — enigma: {' + '.join(env.addends)} = {env.result}")
    print(f"  lettere distinte: {k} ({', '.join(env.letters)}) · "
          f"assegnazioni possibili: {space:,}")
    print(f"  (l'engine NON conosce la soluzione: riceve solo l'errore)\n")

    world = _seed_world(env, pop, seed)
    best_err = None
    solved_at = None

    for _ in range(generations):
        world.step()
        champ = _best(world)
        dig = env.digits_of(champ.substrate)
        total = sum(env.coef[L] * dig[L] for L in env.letters)
        err = abs(total)
        lead_bad = any(dig[L] == 0 for L in env.leading)

        if not lead_bad and (best_err is None or err < best_err):
            best_err = err
            assign = " ".join(f"{L}={dig[L]}" for L in env.letters)
            print(f"  gen {world.generation:4d} | scarto {err:6d} | {assign} | {_render(env, dig)}")

        if champ.beats_sota:
            solved_at = world.generation
            break

    tried = world.archive.stats()["total"]
    print()
    if solved_at is not None:
        dig = env.digits_of(_best(world).substrate)
        print(f"  ✔ RISOLTO alla generazione {solved_at}.")
        print(f"    {_render(env, dig)}")
        print("    " + "  ".join(f"{L}={dig[L]}" for L in env.letters))
        print(f"    Tentativi valutati: {tried:,} su {space:,} assegnazioni. Non e' forza bruta: e' selezione.")
    else:
        dig = env.digits_of(_best(world).substrate)
        print(f"  ✗ Non risolto in {generations} generazioni ({tried:,} tentativi).")
        print(f"    Miglior tentativo (scarto {best_err}): {_render(env, dig)}")
        print(f"    Riprova con piu' generazioni o popolazione.")
    world.archive.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="CHIMERA — risolvere un criptaritmo per evoluzione")
    ap.add_argument("--addends", nargs="+", default=["SEND", "MORE"], help="gli addendi (parole)")
    ap.add_argument("--result", default="MONEY", help="la parola-risultato")
    ap.add_argument("--generations", type=int, default=500)
    ap.add_argument("--pop", type=int, default=300, help="concetti per continente")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.addends, args.result, args.generations, args.pop, args.seed)


if __name__ == "__main__":
    main()
