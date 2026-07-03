"""
substrate.py
============

Un *substrato* è il modo in cui un concetto esiste concretamente.

Principio fondamentale di CHIMERA:
    Nessun concetto ha un significato assegnato dagli esseri umani.
    Il significato emerge SOLO dagli effetti che il concetto produce
    quando viene interrogato dall'ambiente.

Per questo un substrato non "significa" nulla. Espone solo:
    - una struttura mutabile/ricombinabile
    - un metodo express(x) che, dato un input dell'ambiente,
      produce un output. Cosa quell'output "voglia dire" lo decide
      la fitness, non noi.

In v0.1 esiste un solo substrato concreto: ExpressionTree (un piccolo
"programma" matematico). L'astrazione Substrate è pensata perche' domani
si possano aggiungere Vector, Graph, NeuralNet... e persino permettere a un
concetto di CAMBIARE substrato durante l'evoluzione (FASE estrema del manifesto).
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod

import numpy as np


class Substrate(ABC):
    """Forma concreta con cui un concetto esiste."""

    kind: str = "abstract"

    @abstractmethod
    def express(self, x: np.ndarray) -> np.ndarray:
        """Interroga il concetto sull'input dell'ambiente e restituisce un output."""

    @abstractmethod
    def mutate(self, physics: "Physics") -> "Substrate":
        """Restituisce una COPIA mutata (l'originale resta immutato)."""

    @abstractmethod
    def recombine(self, other: "Substrate", physics: "Physics") -> "Substrate":
        """Fonde due substrati in uno nuovo (accoppiamento)."""

    @abstractmethod
    def complexity(self) -> int:
        """Dimensione strutturale, usata come pressione di parsimonia."""

    @abstractmethod
    def describe(self) -> str:
        """Traduzione leggibile SOLO per l'osservatore umano (non usata dal motore)."""

    def learn(self, x: np.ndarray, y: np.ndarray) -> "Substrate":
        """APPRENDIMENTO: adatta i parametri per far somigliare express(x) a y.
        Default: nessun apprendimento in forma chiusa (il concetto impara solo
        per evoluzione). Le forme lineari nel readout (vettore, rete) lo ridefiniscono
        e risolvono ai minimi quadrati. E' il canale 'entro la vita' dei concetti."""
        return self


# --------------------------------------------------------------------------- #
#  Physics: le "regole del continente" in cui vive un substrato.
#  Continenti diversi = matematiche diverse (operatori/costanti differenti).
# --------------------------------------------------------------------------- #

class Physics:
    """Definisce quali operazioni e costanti sono ammesse in un continente."""

    ALL_UNARY = ("neg", "abs", "sin", "cos", "tanh", "log", "sqrt", "square", "exp", "inv")
    ALL_BINARY = ("add", "sub", "mul", "div")

    def __init__(
        self,
        unary: tuple[str, ...] | None = None,
        binary: tuple[str, ...] | None = None,
        const_range: tuple[float, float] = (-3.0, 3.0),
        max_depth: int = 4,
        rng: random.Random | None = None,
        meta_targets: tuple[str, ...] = ("vector", "mlp"),
        n_features: int = 20,
        hidden: int = 14,
        feature_scale: float = 8.0,
        n_vars: int = 1,
    ):
        self.unary = unary if unary is not None else self.ALL_UNARY
        self.binary = binary if binary is not None else self.ALL_BINARY
        self.const_range = const_range
        self.max_depth = max_depth
        self.rng = rng or random.Random()
        # v0.3: metamorfosi. probe_x sono gli input dell'ambiente su cui si
        # distilla il comportamento quando un concetto cambia forma.
        self.meta_targets = meta_targets
        self.n_features = n_features
        self.hidden = hidden
        self.feature_scale = feature_scale     # ampiezza delle frequenze delle feature
        self.n_vars = n_vars                   # numero di variabili di input (>=1)
        self.probe_x: np.ndarray | None = None
        self.np_rng = np.random.default_rng(self.rng.randint(0, 2**31 - 1))

    def attach_probe(self, x: np.ndarray) -> None:
        """Collega gli input dell'ambiente (necessari per la distillazione)."""
        self.probe_x = np.asarray(x, dtype=np.float64)

    def random_const(self) -> float:
        lo, hi = self.const_range
        return round(self.rng.uniform(lo, hi), 4)


# --------------------------------------------------------------------------- #
#  Operatori vettoriali "protetti" (mai NaN/inf che rompono l'evoluzione).
# --------------------------------------------------------------------------- #

def _protect(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a, nan=0.0, posinf=1e6, neginf=-1e6)


def _as2d(X: np.ndarray) -> np.ndarray:
    """Porta l'input a forma (N, D). Un input 1D diventa (N, 1): cosi' i
    substrati gestiscono in modo uniforme problemi a una o piu' variabili."""
    X = np.asarray(X, dtype=np.float64)
    return X[:, None] if X.ndim == 1 else X


UNARY_FN = {
    "neg": lambda a: -a,
    "abs": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tanh": np.tanh,
    "log": lambda a: np.log(np.abs(a) + 1e-9),
    "sqrt": lambda a: np.sqrt(np.abs(a)),
    "square": lambda a: np.clip(a, -1e3, 1e3) ** 2,
    "exp": lambda a: np.exp(np.clip(a, -30, 30)),
    "inv": lambda a: 1.0 / (a + np.where(np.abs(a) < 1e-9, 1e-9, 0.0)),
}

BINARY_FN = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / (b + np.where(np.abs(b) < 1e-9, 1e-9, 0.0)),
}


# --------------------------------------------------------------------------- #
#  ExpressionTree: il concetto-come-programma.
# --------------------------------------------------------------------------- #

class _Node:
    __slots__ = ("op", "value", "children")

    def __init__(self, op: str, value=None, children=None):
        # op in {"var", "const", <unary>, <binary>}
        self.op = op
        self.value = value
        self.children: list[_Node] = children or []

    def clone(self) -> "_Node":
        return _Node(self.op, self.value, [c.clone() for c in self.children])

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def eval(self, X: np.ndarray) -> np.ndarray:
        # X ha forma (N, D); ogni nodo restituisce un vettore (N,)
        if self.op == "var":
            return X[:, int(self.value or 0)]
        if self.op == "const":
            return np.full(X.shape[0], self.value)
        if self.op in UNARY_FN:
            return _protect(UNARY_FN[self.op](self.children[0].eval(X)))
        # binary
        a = self.children[0].eval(X)
        b = self.children[1].eval(X)
        return _protect(BINARY_FN[self.op](a, b))

    def to_str(self) -> str:
        if self.op == "var":
            return f"x{int(self.value or 0)}"
        if self.op == "const":
            return f"{self.value:g}"
        if self.op in UNARY_FN:
            return f"{self.op}({self.children[0].to_str()})"
        sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[self.op]
        return f"({self.children[0].to_str()} {sym} {self.children[1].to_str()})"

    def iter_nodes(self):
        yield self
        for c in self.children:
            yield from c.iter_nodes()


def _random_node(physics: Physics, depth: int) -> _Node:
    rng = physics.rng
    # ai livelli piu' profondi (o per caso) genera una foglia
    if depth <= 0 or (depth < physics.max_depth and rng.random() < 0.3):
        if rng.random() < 0.6:
            return _Node("var", value=rng.randrange(physics.n_vars))
        return _Node("const", value=physics.random_const())

    if rng.random() < 0.5 and physics.unary:
        op = rng.choice(physics.unary)
        return _Node(op, children=[_random_node(physics, depth - 1)])
    op = rng.choice(physics.binary)
    return _Node(op, children=[_random_node(physics, depth - 1), _random_node(physics, depth - 1)])


class ExpressionTree(Substrate):
    kind = "expression"

    def __init__(self, root: _Node):
        self.root = root

    # -- costruzione ------------------------------------------------------- #
    @classmethod
    def random(cls, physics: Physics) -> "ExpressionTree":
        return cls(_random_node(physics, physics.max_depth))

    # -- interfaccia Substrate -------------------------------------------- #
    def express(self, x: np.ndarray) -> np.ndarray:
        return self.root.eval(_as2d(x))

    def mutate(self, physics: Physics) -> "ExpressionTree":
        clone = self.root.clone()
        nodes = list(clone.iter_nodes())
        target = physics.rng.choice(nodes)
        roll = physics.rng.random()

        if target.op == "const" and roll < 0.5:
            # perturbazione di una costante
            target.value = round(target.value + physics.rng.gauss(0, 0.8), 4)
        elif roll < 0.75:
            # sostituzione di sottoalbero (mutazione strutturale)
            new_sub = _random_node(physics, min(3, physics.max_depth))
            target.op, target.value, target.children = (
                new_sub.op, new_sub.value, new_sub.children,
            )
        else:
            # mutazione puntuale dell'operatore, preservando l'arieta'
            if target.op in BINARY_FN and physics.binary:
                target.op = physics.rng.choice(physics.binary)
            elif target.op in UNARY_FN and physics.unary:
                target.op = physics.rng.choice(physics.unary)
            elif target.op == "var":
                target.op, target.value = "const", physics.random_const()
            elif target.op == "const":
                target.op, target.value = "var", physics.rng.randrange(physics.n_vars)
        return ExpressionTree(clone)

    def recombine(self, other: "Substrate", physics: Physics) -> "Substrate":
        # il crossover di sottoalberi ha senso solo tra due alberi; con una
        # forma diversa non c'e' materiale strutturale compatibile -> muta.
        if not isinstance(other, ExpressionTree):
            return self.mutate(physics)
        # crossover di sottoalberi: prendo la struttura di self e vi innesto
        # un sottoalbero di other (come uno scambio di materiale genetico)
        child = self.root.clone()
        child_nodes = list(child.iter_nodes())
        donor_nodes = list(other.root.iter_nodes())
        cut = physics.rng.choice(child_nodes)
        graft = physics.rng.choice(donor_nodes).clone()
        cut.op, cut.value, cut.children = graft.op, graft.value, graft.children
        return ExpressionTree(child)

    def complexity(self) -> int:
        return self.root.size()

    def describe(self) -> str:
        return self.root.to_str()


# --------------------------------------------------------------------------- #
#  VectorFeature: il concetto-come-VETTORE.
#  Un banco di feature non lineari fisse phi(x) + un vettore di pesi w.
#  express(x) = w . phi(x). E' la versione (in scala) dell'idea del manifesto:
#  "un concetto e' un vettore di N dimensioni". Distillabile in forma chiusa.
# --------------------------------------------------------------------------- #

def _rff(x: np.ndarray, freqs: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """Random Fourier features multi-variabile + colonna di bias.
    x -> (N, D); freqs -> (D, F); phases -> (F,). Ritorna (N, F+1)."""
    X = _as2d(x)
    z = np.sin(X @ freqs + phases[None, :])
    return np.hstack([np.ones((len(X), 1)), z])


def _least_squares(phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    w, *_ = np.linalg.lstsq(phi, y, rcond=None)
    return w


class VectorFeature(Substrate):
    kind = "vector"

    def __init__(self, freqs: np.ndarray, phases: np.ndarray, weights: np.ndarray):
        self.freqs = freqs
        self.phases = phases
        self.w = weights

    @classmethod
    def _rand_freqs(cls, physics: Physics):
        F, D = physics.n_features, physics.n_vars
        freqs = physics.np_rng.normal(0, physics.feature_scale, (D, F))
        phases = physics.np_rng.uniform(0, 2 * np.pi, F)
        return freqs, phases

    @classmethod
    def random(cls, physics: Physics) -> "VectorFeature":
        freqs, phases = cls._rand_freqs(physics)
        w = physics.np_rng.normal(0, 0.5, physics.n_features + 1)
        return cls(freqs, phases, w)

    @classmethod
    def distill(cls, x: np.ndarray, y: np.ndarray, physics: Physics) -> "VectorFeature":
        """Costruisce un vettore che RIPRODUCE il comportamento (x -> y)."""
        freqs, phases = cls._rand_freqs(physics)
        w = _least_squares(_rff(x, freqs, phases), y)
        return cls(freqs, phases, w)

    def express(self, x: np.ndarray) -> np.ndarray:
        return _protect(_rff(x, self.freqs, self.phases) @ self.w)

    def mutate(self, physics: Physics) -> "VectorFeature":
        w = self.w.copy()
        freqs, phases = self.freqs.copy(), self.phases.copy()
        if physics.rng.random() < 0.75:
            # perturba i pesi (geometria del vettore)
            mask = physics.np_rng.random(w.shape) < 0.4
            w = w + mask * physics.np_rng.normal(0, 0.3, w.shape)
        else:
            # rimpiazza una feature (cambia la "dimensione" attiva)
            k = physics.rng.randrange(freqs.shape[1])
            freqs[:, k] = physics.np_rng.normal(0, physics.feature_scale, freqs.shape[0])
            phases[k] = physics.np_rng.uniform(0, 2 * np.pi)
        return VectorFeature(freqs, phases, w)

    def recombine(self, other: "Substrate", physics: Physics) -> "Substrate":
        return _behavioral_blend(self, other, physics, VectorFeature)

    def learn(self, x: np.ndarray, y: np.ndarray) -> "VectorFeature":
        # readout lineare: pesi ottimi in forma chiusa, feature fisse
        w = _least_squares(_rff(x, self.freqs, self.phases), y)
        return VectorFeature(self.freqs, self.phases, w)

    def complexity(self) -> int:
        return len(self.w)

    def describe(self) -> str:
        head = ", ".join(f"{v:+.2f}" for v in self.w[:6])
        return f"vector[{len(self.w)}]·rff({head}{'…' if len(self.w) > 6 else ''})"


# --------------------------------------------------------------------------- #
#  TinyMLP: il concetto-come-RETE NEURALE (1 strato nascosto, input 1D).
#  express(x) = tanh(x·W1 + b1) · W2 + b2. Distillabile fissando lo strato
#  nascosto (random features) e risolvendo il readout ai minimi quadrati.
# --------------------------------------------------------------------------- #

class TinyMLP(Substrate):
    kind = "mlp"

    def __init__(self, W1, b1, W2, b2):
        self.W1 = W1      # (H,)
        self.b1 = b1      # (H,)
        self.W2 = W2      # (H,)
        self.b2 = b2      # scalar

    @classmethod
    def random(cls, physics: Physics) -> "TinyMLP":
        H, D = physics.hidden, physics.n_vars
        r = physics.np_rng
        s = physics.feature_scale
        return cls(r.normal(0, s, (D, H)), r.uniform(-s, s, H), r.normal(0, 0.5, H), 0.0)

    @classmethod
    def distill(cls, x: np.ndarray, y: np.ndarray, physics: Physics) -> "TinyMLP":
        H, D = physics.hidden, physics.n_vars
        r = physics.np_rng
        s = physics.feature_scale
        W1 = r.normal(0, s, (D, H))
        b1 = r.uniform(-s, s, H)
        hdn = np.tanh(_as2d(x) @ W1 + b1[None, :])            # (N, H)
        read = _least_squares(np.hstack([hdn, np.ones((len(hdn), 1))]), y)
        return cls(W1, b1, read[:H], float(read[H]))

    def express(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(_as2d(x) @ self.W1 + self.b1[None, :])
        return _protect(h @ self.W2 + self.b2)

    def mutate(self, physics: Physics) -> "TinyMLP":
        r = physics.np_rng
        W1, b1, W2, b2 = self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2
        which = physics.rng.random()
        if which < 0.6:
            m = r.random(W2.shape) < 0.5
            W2 = W2 + m * r.normal(0, 0.25, W2.shape)
            b2 = b2 + r.normal(0, 0.1)
        else:
            mw = r.random(W1.shape) < 0.4
            W1 = W1 + mw * r.normal(0, 0.25, W1.shape)
            mb = r.random(b1.shape) < 0.4
            b1 = b1 + mb * r.normal(0, 0.25, b1.shape)
        return TinyMLP(W1, b1, W2, b2)

    def recombine(self, other: "Substrate", physics: Physics) -> "Substrate":
        return _behavioral_blend(self, other, physics, TinyMLP)

    def learn(self, x: np.ndarray, y: np.ndarray) -> "TinyMLP":
        # strato nascosto fisso (evoluto), readout ottimo ai minimi quadrati
        H = self.W1.shape[1]
        hdn = np.tanh(_as2d(x) @ self.W1 + self.b1[None, :])
        read = _least_squares(np.hstack([hdn, np.ones((len(hdn), 1))]), y)
        return TinyMLP(self.W1, self.b1, read[:H], float(read[H]))

    def complexity(self) -> int:
        return self.W1.shape[1]

    def describe(self) -> str:
        return f"mlp({self.W1.shape[0]}→{self.W1.shape[1]}→1, |w|={np.abs(self.W2).sum():.2f})"


# --------------------------------------------------------------------------- #
#  Ricombinazione comportamentale e METAMORFOSI.
#  Entrambe usano solo express(): funzionano quindi anche tra forme diverse.
# --------------------------------------------------------------------------- #

def _behavioral_blend(a: Substrate, b: Substrate, physics: Physics, target_cls):
    """Figlio della forma di 'a' che riproduce la MEDIA dei comportamenti dei genitori."""
    x = physics.probe_x
    if x is None:
        return a.mutate(physics)
    y = 0.5 * (a.express(x) + b.express(x))
    return target_cls.distill(x, y, physics)


_META_BUILDERS = {"vector": VectorFeature, "mlp": TinyMLP}


def metamorphose(substrate: Substrate, physics: Physics) -> Substrate | None:
    """Cambia la FORMA del concetto preservandone il comportamento (distillazione).

    Ritorna None se la metamorfosi non e' possibile (manca probe_x o nessuna
    forma bersaglio diversa da quella attuale). 'expression' non e' un bersaglio
    di distillazione (un programma non si adatta in forma chiusa): i programmi
    nascono per genesi e possono trasformarsi in vettore/rete, non viceversa.
    """
    x = physics.probe_x
    if x is None:
        return None
    targets = [t for t in physics.meta_targets if t != substrate.kind and t in _META_BUILDERS]
    if not targets:
        return None
    target = physics.rng.choice(targets)
    y = substrate.express(x)
    if not np.all(np.isfinite(y)):
        return None
    return _META_BUILDERS[target].distill(x, y, physics)
