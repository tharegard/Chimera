"""
feynman.py
==========

Benchmark scientifico: le equazioni di Feynman (regressione simbolica).

Compito: dati campioni (variabili fisiche -> risultato) di una legge fisica NOTA,
un concetto deve riscoprire la relazione. E' un banco di prova standard e onesto
per la regressione simbolica.

Due scelte per l'ONESTA' del risultato:
  1. TRAIN/TEST SPLIT. Si allena su una parte dei dati e si MISURA su dati mai
     visti. Cosi' distinguiamo la scoperta dalla memorizzazione (overfitting).
  2. BASELINE. La regressione lineare (standard) e' lo "stato dell'arte" da battere.

Gli input hanno scale molto diverse (masse, distanze, ...): vengono standardizzati
per colonna, e anche il bersaglio, cosi' tutte le forme (vettore/rete/programma)
lavorano su scale confrontabili. Le metriche riportate (R^2 di test) sono comunque
indipendenti dalla scala.
"""

from __future__ import annotations

import itertools

import numpy as np

from ..core.concept import Concept
from .base import Environment

# nome -> (variabili, intervalli di campionamento, funzione, formula leggibile)
EQUATIONS: dict[str, tuple] = {
    "prodotto":    (["a", "b"],             [(1, 5), (1, 5)],
                    lambda X: X[:, 0] * X[:, 1], "a·b"),
    "rapporto":    (["a", "b"],             [(1, 5), (1, 5)],
                    lambda X: X[:, 0] / X[:, 1], "a/b"),
    "cinetica":    (["m", "v"],             [(1, 5), (1, 5)],
                    lambda X: 0.5 * X[:, 0] * X[:, 1] ** 2, "½·m·v²"),
    "gravitazione": (["m1", "m2", "r"],     [(1, 5), (1, 5), (1, 5)],
                    lambda X: X[:, 0] * X[:, 1] / X[:, 2] ** 2, "m1·m2/r²"),
    "gaussiana":   (["t"],                  [(-3, 3)],
                    lambda X: np.exp(-X[:, 0] ** 2 / 2) / np.sqrt(2 * np.pi), "e^(−t²/2)/√(2π)"),
    "distanza":    (["x1", "y1", "x2", "y2"], [(0, 5)] * 4,
                    lambda X: np.sqrt((X[:, 2] - X[:, 0]) ** 2 + (X[:, 3] - X[:, 1]) ** 2),
                    "√((x2−x1)²+(y2−y1)²)"),
    "relativita":  (["m", "v", "c"],        [(1, 5), (1, 2), (3, 10)],
                    lambda X: X[:, 0] / np.sqrt(1 - (X[:, 1] / X[:, 2]) ** 2), "m/√(1−v²/c²)"),
    "pendolo":     (["L", "g"],             [(1, 5), (9, 10)],
                    lambda X: 2 * np.pi * np.sqrt(X[:, 0] / X[:, 1]), "2π·√(L/g)"),
    "smorzamento": (["A", "x"],             [(1, 5), (0, 3)],
                    lambda X: X[:, 0] * np.exp(-X[:, 1]), "A·e^(−x)"),
    "onda":        (["A", "k", "x"],        [(1, 5), (1, 3), (0, 6)],
                    lambda X: X[:, 0] * np.sin(X[:, 1] * X[:, 2]), "A·sin(k·x)"),
}


def _nrmse(pred, y, scale):
    return float(np.sqrt(np.mean((pred - y) ** 2)) / (scale + 1e-12))


def _r2(pred, y):
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def _poly_features(Xs: np.ndarray, degree: int = 2) -> np.ndarray:
    """Feature polinomiali fino a `degree` (bias + lineari + prodotti/quadrati)."""
    n, d = Xs.shape
    cols = [np.ones(n)]
    for deg in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(d), deg):
            cols.append(np.prod(Xs[:, combo], axis=1))
    return np.stack(cols, axis=1)


class FeynmanEnvironment(Environment):
    def __init__(self, key: str, n_train: int = 400, n_test: int = 400,
                 seed: int = 0, noise: float = 0.0, extrapolate: bool = False):
        self.key = key
        self.var_names, ranges, fn, self.formula = EQUATIONS[key]
        self.name = f"feynman:{key}"
        self.n_vars = len(self.var_names)
        rng = np.random.default_rng(seed)

        # estrapolazione: train sul 70% basso di ogni intervallo, test sul 30% alto
        # (regioni DISGIUNTE, dominio sicuro): misura se la legge generalizza fuori
        # dai dati visti, non se interpola soltanto.
        def split_ranges(frac_lo, frac_hi):
            out = []
            for lo, hi in ranges:
                span = hi - lo
                out.append((lo + frac_lo * span, lo + frac_hi * span))
            return out

        tr_ranges = split_ranges(0.0, 0.7) if extrapolate else ranges
        te_ranges = split_ranges(0.7, 1.0) if extrapolate else ranges

        def sample(n, rgs):
            cols = [rng.uniform(lo, hi, n) for (lo, hi) in rgs]
            X = np.stack(cols, axis=1)
            return X, fn(X).astype(np.float64)

        Xtr, ytr_clean = sample(n_train, tr_ranges)
        Xte, yte = sample(n_test, te_ranges)               # il TEST resta pulito
        # rumore aggiunto SOLO al train: misuriamo se la legge vera si recupera
        ytr = ytr_clean + noise * (np.std(ytr_clean) + 1e-9) * rng.standard_normal(n_train)

        # standardizzazione (statistiche SOLO dal train)
        self._xmu, self._xsd = Xtr.mean(0), Xtr.std(0) + 1e-9
        self._ymu, self._ysd = float(ytr.mean()), float(ytr.std()) + 1e-9
        self._ytr_scale = float(ytr.std()) + 1e-9

        self.x = (Xtr - self._xmu) / self._xsd            # input standardizzato (train)
        self.y = ytr
        self._Xte = (Xte - self._xmu) / self._xsd
        self._yte = yte

        # baseline 1: regressione lineare
        self._lin_w = np.linalg.lstsq(np.hstack([self.x, np.ones((n_train, 1))]), ytr, rcond=None)[0]
        self._lin_train_nrmse = _nrmse(self._lin_pred(self.x), ytr, self._ytr_scale)
        self._lin_test_r2 = _r2(self._lin_pred(self._Xte), yte)
        # baseline 2: regressione polinomiale (grado 2) - piu' forte e onesta
        Ptr, Pte = _poly_features(self.x), _poly_features(self._Xte)
        self._poly_w = np.linalg.lstsq(Ptr, ytr, rcond=None)[0]
        self._poly_test_r2 = _r2(Pte @ self._poly_w, yte)

    def _lin_pred(self, Xs):
        return np.hstack([Xs, np.ones((len(Xs), 1))]) @ self._lin_w

    # -- il concetto predice il risultato (unita' del bersaglio) ---------- #
    def predict(self, substrate, x: np.ndarray) -> np.ndarray:
        return self._ymu + self._ysd * np.nan_to_num(substrate.express(x))

    def learn_target(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x, (self.y - self._ymu) / self._ysd

    # -- valutazione (sul TRAIN) ------------------------------------------ #
    def evaluate(self, concept: Concept) -> tuple[float, bool]:
        try:
            pred = self.predict(concept.substrate, self.x)
        except Exception:
            return float("-inf"), False
        if pred.shape != self.y.shape or not np.all(np.isfinite(pred)):
            return float("-inf"), False
        nrmse = _nrmse(pred, self.y, self._ytr_scale)
        return -nrmse, nrmse < self._lin_train_nrmse

    def sota_score(self) -> float:
        return -self._lin_train_nrmse

    # -- metriche sul TEST (dati mai visti) ------------------------------- #
    def test_metrics(self, concept: Concept) -> dict:
        pred = self.predict(concept.substrate, self._Xte)
        return {
            "test_r2": _r2(pred, self._yte),
            "test_nrmse": _nrmse(pred, self._yte, np.std(self._yte)),
            "lin_test_r2": self._lin_test_r2,
            "poly_test_r2": self._poly_test_r2,
        }

    def curve_labels(self) -> tuple[str, str]:
        return ("valore vero", "predizione del concetto")


class FeynmanScientific(Environment):
    """CHIMERA-S: evolve SPIEGAZIONI, non approssimatori.

    Differenze rispetto a FeynmanEnvironment (CHIMERA-P):
      - input GREZZI (non standardizzati): un programma tipo 1/r² estrapola davvero;
      - solo un readout AFFINE (a*prog+b, 2 parametri) per la scala -> nessuna
        capacita' extra per memorizzare; il concetto deve trovare la FORMA giusta;
      - fitness = accuratezza + SEMPLICITA' (parsimonia forte): premia la legge
        piu' semplice che spiega i dati, non il fit piu' aderente.

    Va usato con continenti di soli PROGRAMMI (meta_targets vuoto): niente forme
    vettore/rete, che interpolano. Lo split e' sempre di estrapolazione
    (train 70% basso, test 30% alto) perche' e' li' che si misura la scoperta.
    """

    def __init__(self, key: str, n_train: int = 400, n_test: int = 400,
                 seed: int = 0, parsimony: float = 0.01):
        self.key = key
        self.var_names, ranges, fn, self.formula = EQUATIONS[key]
        self.name = f"feynman-S:{key}"
        self.n_vars = len(self.var_names)
        self.parsimony = parsimony
        rng = np.random.default_rng(seed)

        def rng_split(frac_lo, frac_hi):
            return [(lo + frac_lo * (hi - lo), lo + frac_hi * (hi - lo)) for lo, hi in ranges]

        def sample(n, rgs):
            X = np.stack([rng.uniform(lo, hi, n) for lo, hi in rgs], axis=1)
            return X, fn(X).astype(np.float64)

        self.x, self.y = sample(n_train, rng_split(0.0, 0.7))     # input GREZZI
        self._Xte, self._yte = sample(n_test, rng_split(0.7, 1.0))
        self._yscale = float(self.y.std()) + 1e-9

        # baseline in estrapolazione (lineare e polinomiale, su input grezzi)
        self._lin_w = np.linalg.lstsq(np.hstack([self.x, np.ones((n_train, 1))]), self.y, rcond=None)[0]
        self._lin_nrmse = _nrmse(np.hstack([self.x, np.ones((n_train, 1))]) @ self._lin_w, self.y, self._yscale)
        self._lin_test_r2 = _r2(np.hstack([self._Xte, np.ones((n_test, 1))]) @ self._lin_w, self._yte)
        Ptr, Pte = _poly_features(self.x), _poly_features(self._Xte)
        pw = np.linalg.lstsq(Ptr, self.y, rcond=None)[0]
        self._poly_test_r2 = _r2(Pte @ pw, self._yte)

    def _affine(self, substrate):
        """Adatta la sola scala (a, b) del programma sul train: 2 parametri."""
        prog = np.nan_to_num(substrate.express(self.x))
        A = np.vstack([prog, np.ones_like(prog)]).T
        ab, *_ = np.linalg.lstsq(A, self.y, rcond=None)
        return float(ab[0]), float(ab[1])

    def predict(self, substrate, x: np.ndarray) -> np.ndarray:
        a, b = self._affine(substrate)
        return a * np.nan_to_num(substrate.express(x)) + b

    def learn_target(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x, self.y

    def evaluate(self, concept: Concept) -> tuple[float, bool]:
        try:
            a, b = self._affine(concept.substrate)
            pred = a * np.nan_to_num(concept.substrate.express(self.x)) + b
        except Exception:
            return float("-inf"), False
        if pred.shape != self.y.shape or not np.all(np.isfinite(pred)):
            return float("-inf"), False
        nrmse = _nrmse(pred, self.y, self._yscale)
        # accuratezza + semplicita': la legge piu' semplice che spiega i dati
        fitness = -nrmse - self.parsimony * concept.substrate.complexity()
        return fitness, nrmse < self._lin_nrmse

    def sota_score(self) -> float:
        return -self._lin_nrmse

    def test_metrics(self, concept: Concept) -> dict:
        pred = self.predict(concept.substrate, self._Xte)
        return {
            "test_r2": _r2(pred, self._yte),
            "lin_test_r2": self._lin_test_r2,
            "poly_test_r2": self._poly_test_r2,
        }

    def curve_labels(self) -> tuple[str, str]:
        return ("valore vero", "legge del concetto")
