"""
symbolic_regression.py
======================

Primo ambiente concreto di CHIMERA.

Compito: dato un input x, produrre un output che approssimi una funzione
bersaglio *nascosta*. Il concetto non sa cos'e' la funzione: viene solo
misurato su quanto i suoi effetti (gli output) somigliano a quelli attesi.

Stato dell'arte (baseline): la migliore retta y = a*x + b ai minimi quadrati.
Un concetto "beats_sota" solo se fa meglio di quella retta.

Questo e' volutamente un problema in cui l'ottimizzazione tradizionale
(regressione lineare) e' facilmente battibile da una struttura non lineare:
serve a verificare che il motore evolutivo trovi davvero strutture diverse.
"""

from __future__ import annotations

import numpy as np

from ..core.concept import Concept
from .base import Environment


class SymbolicRegression(Environment):
    name = "symbolic_regression"

    def __init__(
        self,
        target=lambda x: np.sin(2.0 * x) + 0.3 * x * x,
        x_range: tuple[float, float] = (-3.0, 3.0),
        n_points: int = 128,
        parsimony: float = 0.002,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.x = np.sort(rng.uniform(x_range[0], x_range[1], n_points)).astype(np.float64)
        self.y = target(self.x).astype(np.float64)
        self.parsimony = parsimony
        self._sota_rmse = self._linear_baseline_rmse()

    # -- stato dell'arte --------------------------------------------------- #
    def _linear_baseline_rmse(self) -> float:
        a, b = np.polyfit(self.x, self.y, 1)
        pred = a * self.x + b
        return float(np.sqrt(np.mean((pred - self.y) ** 2)))

    def sota_score(self) -> float:
        # fitness equivalente della baseline (senza penalita' di complessita')
        return -self._sota_rmse

    # -- valutazione ------------------------------------------------------- #
    def evaluate(self, concept: Concept) -> tuple[float, bool]:
        try:
            pred = concept.substrate.express(self.x)
        except Exception:
            return float("-inf"), False

        if pred.shape != self.y.shape or not np.all(np.isfinite(pred)):
            return float("-inf"), False

        rmse = float(np.sqrt(np.mean((pred - self.y) ** 2)))
        penalty = self.parsimony * concept.substrate.complexity()
        fitness = -rmse - penalty
        beats = rmse < self._sota_rmse
        return fitness, beats

    # -- suggerimenti per la visualizzazione ------------------------------ #
    def hist_range(self) -> tuple[float, float]:
        return (-1.5, 0.0)

    def curve_labels(self) -> tuple[str, str]:
        return ("funzione bersaglio", "predizione del miglior concetto")

    def summarize(self, concept: Concept, lineage=None) -> dict:
        x, y = self.x, self.y
        pred = np.nan_to_num(concept.substrate.express(x))
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        base = float(self._sota_rmse)
        improv = base / rmse if rmse > 1e-9 else float("inf")
        corr = 0.0 if np.std(pred) < 1e-9 or np.std(y) < 1e-9 else float(np.corrcoef(pred, y)[0, 1])
        n_osc = _turning_points(pred)
        n_osc_t = _turning_points(y)
        worst_x = float(x[int(np.argmax(np.abs(pred - y)))])

        bullets = [
            f"È {improv:.1f}× più preciso della migliore retta (errore {rmse:.3f} contro "
            f"{base:.3f}), con una corrispondenza del {max(0.0, corr) * 100:.0f}% al bersaglio.",
        ]
        if concept.substrate_kind == "expression":
            bullets.append(f"Calcola questa espressione: {concept.describe()}")
        else:
            bullets.append(f"È una funzione non lineare appresa ({concept.describe()}): non ha una "
                           f"formula leggibile, ma il suo comportamento è la curva verde.")
        bullets.append(f"Ha {n_osc} cambi di pendenza (il bersaglio ne ha {n_osc_t}): "
                       + ("cattura le oscillazioni che una retta non può rappresentare."
                          if n_osc >= 2 else "segue l'andamento generale del bersaglio."))
        sym = _symmetry(concept.substrate, x)
        if sym != "nessuna":
            bullets.append(f"Ha scoperto una simmetria: è una funzione {sym}.")
        bullets.append(f"Dove sbaglia di più: attorno a x = {worst_x:+.2f}.")
        return {
            "headline": f"Il campione batte lo stato dell'arte {improv:.1f}×.",
            "surprise": ("Sorprendente: partito da programmi, ha trovato una struttura non ovvia che "
                         "nessuna ottimizzazione lineare avrebbe prodotto." if improv > 1.5 else
                         "Per ora non batte in modo netto le soluzioni tradizionali."),
            "bullets": bullets,
        }


def _turning_points(v: np.ndarray) -> int:
    d = np.diff(v)
    d = d[np.abs(d) > 1e-6]
    if len(d) < 2:
        return 0
    return int(np.sum(np.diff(np.sign(d)) != 0))


def _symmetry(substrate, x: np.ndarray) -> str:
    px, pnx = substrate.express(x), substrate.express(-x)
    def corr(a, b):
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    if corr(px, pnx) > 0.97:
        return "pari (simmetrica)"
    if corr(px, -pnx) > 0.97:
        return "dispari (antisimmetrica)"
    return "nessuna"
