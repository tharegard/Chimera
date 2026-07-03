"""
compression.py
==============

Ambiente REALE: compressione dati senza perdita, per predizione.

Idea (codifica predittiva). Un segnale d[0..N-1] va trasmesso. Un concetto e'
un MODELLO che predice il valore in ogni posizione. Si trasmettono allora solo:
    - il concetto stesso (i suoi parametri = il "codice" del compressore)
    - i residui r = d - predizione, codificati entropicamente.
Meglio il modello predice, piu' piccoli i residui, meno bit servono.
Il concetto NON e' un accessorio del compressore: il concetto E' il compressore.

Due accorgimenti perche' l'evoluzione possa davvero imparare:
  1. SCALA. Il concetto modella il segnale STANDARDIZZATO (media 0, scala 1):
     cosi' i suoi output naturali (~[-1,1]) hanno la stessa scala del bersaglio.
     La coppia (media, scala) fa parte del codice (costo fisso, poche decine di bit).
  2. GRADIENTE LISCIO. Il costo in bit dei residui usa la stima gaussiana
     0.5*log2(2*pi*e*sigma^2): e' continua e premia OGNI riduzione di varianza,
     dando alla selezione una salita da scalare (codifica predittiva gaussiana,
     un'idealizzazione standard di cio' che un coder aritmetico adattivo ottiene).

Stato dell'arte (baseline): il migliore tra il costo del segnale grezzo e quello
delle differenze d[i]-d[i-1] (delta coding), con lo stesso stimatore. Un concetto
sopravvive solo se comprime MEGLIO di questo.

Fitness = rapporto di compressione = bit_grezzi / bit_totali (piu' alto = meglio).
"""

from __future__ import annotations

import numpy as np

from ..core.concept import Concept
from .base import Environment

_BITS_PER_PARAM = 16.0        # costo per descrivere un parametro del modello
_HEADER_BITS = 32.0          # costo di (media, scala) del segnale
_LEVELS = 256                # segnale a 8 bit (0..255)


def _default_signal(n: int) -> np.ndarray:
    """Segnale strutturato tipo sensore/onda: periodico + trend, quantizzato 8 bit."""
    i = np.arange(n)
    s = (128
         + 55 * np.sin(2 * np.pi * 3 * i / n)
         + 28 * np.sin(2 * np.pi * 7 * i / n + 1.0)
         + 22 * (i / n)
         + 8 * np.sin(2 * np.pi * 13 * i / n + 0.5))
    return np.clip(np.round(s), 0, _LEVELS - 1)


def _gaussian_bits(residual: np.ndarray) -> float:
    """Bit totali per codificare i residui, stima gaussiana (+1/12 = quantizzazione)."""
    var = float(np.mean(residual ** 2))
    bpp = 0.5 * np.log2(2 * np.pi * np.e * (var + 1.0 / 12.0))
    return len(residual) * max(0.0, bpp)


class PredictiveCompression(Environment):
    name = "compression"

    def __init__(self, signal: np.ndarray | None = None, n: int = 256):
        self.d = (_default_signal(n) if signal is None else
                  np.clip(np.round(signal), 0, _LEVELS - 1)).astype(np.float64)
        n = len(self.d)
        self.x = (np.arange(n) / (n - 1)) * 6.0 - 3.0     # posizioni in [-3, 3]
        self.y = self.d                                    # la "verita'": il segnale
        self._mu = float(self.d.mean())
        self._sd = float(self.d.std()) or 1.0

        self._raw_bits = _gaussian_bits(self.d - self._mu)
        self._delta_bits = _gaussian_bits(np.diff(self.d, prepend=self.d[0])) + _HEADER_BITS
        self._baseline_bits = min(self._raw_bits, self._delta_bits)
        self._baseline_kind = "grezzo" if self._raw_bits <= self._delta_bits else "delta"

    # -- il concetto predice il segnale (in unita' del segnale) ----------- #
    def predict(self, substrate, x: np.ndarray) -> np.ndarray:
        q = np.nan_to_num(substrate.express(x))            # modello standardizzato
        return self._mu + self._sd * q

    def learn_target(self) -> tuple[np.ndarray, np.ndarray]:
        # il substrato modella il segnale STANDARDIZZATO
        return self.x, (self.d - self._mu) / self._sd

    # -- costo in bit di un concetto-compressore -------------------------- #
    def _total_bits(self, concept: Concept) -> float:
        try:
            pred = self.predict(concept.substrate, self.x)
        except Exception:
            return float("inf")
        if pred.shape != self.d.shape or not np.all(np.isfinite(pred)):
            return float("inf")
        residual = self.d - np.clip(pred, -_LEVELS, 2 * _LEVELS)
        model_bits = concept.substrate.complexity() * _BITS_PER_PARAM + _HEADER_BITS
        return _gaussian_bits(residual) + model_bits

    def evaluate(self, concept: Concept) -> tuple[float, bool]:
        total = self._total_bits(concept)
        if not np.isfinite(total) or total <= 0:
            return 0.0, False
        return self._raw_bits / total, total < self._baseline_bits

    def sota_score(self) -> float:
        return self._raw_bits / self._baseline_bits

    # -- suggerimenti per la visualizzazione ------------------------------ #
    def hist_range(self) -> tuple[float, float]:
        return (0.0, max(4.0, self.sota_score() * 2.0))

    def curve_labels(self) -> tuple[str, str]:
        return ("segnale reale", "modello del compressore")

    def summarize(self, concept: Concept, lineage=None) -> dict:
        total = self._total_bits(concept)
        n = len(self.d)
        ratio = self._raw_bits / total if np.isfinite(total) and total > 0 else 0.0
        base_ratio = self.sota_score()
        pred = np.clip(self.predict(concept.substrate, self.x), -_LEVELS, 2 * _LEVELS)
        res_bpp = _gaussian_bits(self.d - pred) / n
        raw_bpp = self._raw_bits / n
        model_bits = concept.substrate.complexity() * _BITS_PER_PARAM + _HEADER_BITS
        beats = total < self._baseline_bits

        bullets = [
            f"Comprime il segnale {ratio:.2f}× (da {raw_bpp:.2f} a {total / n:.2f} bit per campione).",
            f"Baseline classica ({self._baseline_kind} + entropia): {base_ratio:.2f}×. "
            + ("Il concetto la BATTE." if beats else "Il concetto non la batte ancora."),
            f"Dopo aver tolto il modello restano {res_bpp:.2f} bit/campione di residuo "
            f"(più è basso, meglio il modello ha capito il segnale).",
            f"Il modello costa {model_bits:.0f} bit da descrivere "
            f"({concept.substrate.complexity()} parametri + media/scala).",
        ]
        return {
            "headline": f"Comprime il segnale {ratio:.2f}× "
                        + ("(batte la baseline classica)." if beats else "(sotto la baseline)."),
            "surprise": ("Interessante: un modello evoluto libero comprime meglio di un "
                         "compressore classico a delta." if beats else
                         "Per ora i metodi classici restano avanti su questo segnale."),
            "bullets": bullets,
        }
