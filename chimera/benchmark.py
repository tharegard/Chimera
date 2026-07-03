"""
benchmark.py
============

Benchmark scientifico di CHIMERA sulle equazioni di Feynman.

Per ogni legge fisica: evolve una popolazione sul TRAIN e misura il concetto
migliore sul TEST (dati mai visti). Riporta una tabella onesta:
    - R^2 di test del concetto CHIMERA
    - R^2 di test della baseline (regressione lineare)
    - se CHIMERA batte la baseline, e se ha "risolto" (R^2 > 0.99)

Uso:
    python -m chimera.benchmark --generations 80 --pop 120
    python -m chimera.benchmark --eq gravitazione --generations 120
"""

from __future__ import annotations

import argparse
import random
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from .core.archive import Archive
from .core.evolution import Continent, World
from .core.substrate import Physics
from .environments.feynman import EQUATIONS, FeynmanEnvironment, FeynmanScientific

# operatori ricchi (fisica): tutti i continenti, ampiezza feature moderata
_CONT = [
    ("Alpha", ("neg", "abs", "sin", "cos", "sqrt", "square")),
    ("Beta",  ("tanh", "square", "sqrt", "neg", "abs", "inv")),
    ("Gamma", ("log", "sqrt", "abs", "square", "sin", "exp")),
    ("Delta", ("sin", "cos", "tanh", "neg", "abs", "sqrt", "square", "exp", "inv")),
]


def _continents(pop, n_vars, seed, scientific=False):
    conts = []
    for i, (name, unary) in enumerate(_CONT):
        # CHIMERA-S: solo programmi (meta_targets vuoto -> niente vettore/rete),
        # alberi piu' profondi per comporre leggi.
        physics = Physics(unary=unary, binary=("add", "sub", "mul", "div"),
                          max_depth=5 if scientific else 4, rng=random.Random(seed + i + 1),
                          n_vars=n_vars, feature_scale=3.0,
                          meta_targets=() if scientific else ("vector", "mlp"))
        conts.append(Continent(name, physics, World.seed_population(physics, name, pop)))
    return conts


def run_one(key, generations, pop, seed, n, noise, extrapolate, scientific):
    if scientific:
        env = FeynmanScientific(key, n_train=n, n_test=n, seed=seed)
    else:
        env = FeynmanEnvironment(key, n_train=n, n_test=n, seed=seed, noise=noise,
                                 extrapolate=extrapolate)
    archive = Archive(":memory:", run_id=key)
    world = World(env, _continents(pop, env.n_vars, seed, scientific), archive, seed=seed)
    world.bootstrap()
    for _ in range(generations):
        world.step()
    champ = max((c for cont in world.continents for c in cont.population),
                key=lambda c: c.fitness)
    m = env.test_metrics(champ)
    archive.close()
    return {
        "form": champ.substrate_kind,
        "test_r2": m["test_r2"], "lin_r2": m["lin_test_r2"], "poly_r2": m["poly_test_r2"],
    }


def aggregate(key, generations, pop, seeds, n, noise, extrapolate, scientific):
    """Ripete su piu' semi e riassume (media +/- dev)."""
    runs = [run_one(key, generations, pop, s, n, noise, extrapolate, scientific) for s in range(seeds)]
    r2 = np.array([x["test_r2"] for x in runs])
    lin = float(np.mean([x["lin_r2"] for x in runs]))
    poly = float(np.mean([x["poly_r2"] for x in runs]))
    best_base = max(lin, poly)
    var_names, _, _, formula = EQUATIONS[key]
    return {
        "key": key, "formula": formula, "vars": len(var_names),
        "form": runs[-1]["form"], "r2_mean": float(r2.mean()), "r2_std": float(r2.std()),
        "lin": lin, "poly": poly,
        "beats": r2.mean() > best_base + 1e-4,
        "solved": float(np.mean(r2 > 0.99)),   # frazione di semi risolti
    }


def main():
    ap = argparse.ArgumentParser(description="CHIMERA - benchmark Feynman (regressione simbolica)")
    ap.add_argument("--generations", type=int, default=80)
    ap.add_argument("--pop", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=1, help="numero di semi (ripetibilita')")
    ap.add_argument("--n", type=int, default=400, help="campioni per train e per test")
    ap.add_argument("--noise", type=float, default=0.0, help="rumore sul train (frazione di sigma)")
    ap.add_argument("--extrapolate", action="store_true",
                    help="train sul 70%% basso degli intervalli, test sul 30%% alto (estrapolazione)")
    ap.add_argument("--scientific", action="store_true",
                    help="CHIMERA-S: evolve leggi (programmi grezzi + parsimonia), split di estrapolazione")
    ap.add_argument("--eq", default=None, help="una sola equazione (default: tutte)")
    args = ap.parse_args()
    if args.scientific:
        args.extrapolate = True    # S si misura sempre in estrapolazione

    keys = [args.eq] if args.eq else list(EQUATIONS)

    modo = "CHIMERA-S (leggi)" if args.scientific else "CHIMERA-P (predittivo)"
    print("=" * 92)
    print(f" {modo} - benchmark Feynman  (R^2 su TEST, dati mai visti)")
    print(f" generazioni={args.generations}  pop={args.pop}  semi={args.seeds}"
          f"  campioni={args.n}  rumore={args.noise}  estrapolazione={args.extrapolate}")
    print("=" * 92)
    print(f" {'equazione':<13}{'formula':<24}{'var':>4}  {'forma':<8}{'R2 CHIMERA':>15}"
          f"{'R2 lin':>8}{'R2 poly':>9}  esito")
    print("-" * 92)

    rows = []
    for k in keys:
        r = aggregate(k, args.generations, args.pop, args.seeds, args.n, args.noise,
                      args.extrapolate, args.scientific)
        rows.append(r)
        chi = f"{r['r2_mean']:.4f}" + (f"±{r['r2_std']:.3f}" if args.seeds > 1 else "")
        base = max(r["lin"], r["poly"])
        esito = "RISOLTO" if r["r2_mean"] > 0.99 else ("batte baseline" if r["r2_mean"] > base + 1e-4 else "-")
        print(f" {r['key']:<13}{r['formula']:<24}{r['vars']:>4}  {r['form']:<8}{chi:>15}"
              f"{r['lin']:>8.3f}{r['poly']:>9.3f}  {esito}")

    print("-" * 92)
    solved = sum(r["r2_mean"] > 0.99 for r in rows)
    beats = sum(r["beats"] for r in rows)
    print(f" Risolte (R^2>0.99): {solved}/{len(rows)}   |   "
          f"Battono il MIGLIOR baseline (lineare o polinomiale): {beats}/{len(rows)}")


if __name__ == "__main__":
    main()
