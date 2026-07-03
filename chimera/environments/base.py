"""
base.py
=======

Un Ambiente mette alla prova un concetto e ne misura SOLO gli effetti.

Contratto minimo:
    evaluate(concept) -> (fitness: float, beats_sota: bool)
    sota_score()      -> float   (fitness dello stato dell'arte, per riferimento)

L'ambiente e' l'unica autorita' che assegna significato: un concetto "vale"
esattamente quanto vale il risultato che produce. Include una nozione di
"stato dell'arte" (baseline): un concetto sopravvive davvero solo se lo batte.

Ogni ambiente espone anche self.x / self.y (input e "verita'" dell'ambiente),
usati dalla visualizzazione (curva e atlante). I metodi opzionali piu' in basso
(hist_range, curve_labels, summarize) permettono a ciascun ambiente di guidare
la dashboard e il traduttore SENZA che il resto del sistema sappia di quale
ambiente si tratti.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.concept import Concept


class Environment(ABC):
    name: str = "abstract"
    x: np.ndarray
    y: np.ndarray

    @abstractmethod
    def evaluate(self, concept: Concept) -> tuple[float, bool]:
        """Restituisce (fitness, beats_sota)."""

    @abstractmethod
    def sota_score(self) -> float:
        """Fitness dello stato dell'arte, per riferimento a schermo."""

    def predict(self, substrate, x: np.ndarray) -> np.ndarray:
        """La predizione del concetto nelle stesse unita' di self.y.
        Di default e' l'output grezzo del substrato; un ambiente puo' ridefinirla
        (es. compressione: riscala il modello standardizzato al segnale)."""
        return substrate.express(x)

    def learn_target(self) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) che l'output del substrato dovrebbe imitare quando un concetto
        APPRENDE. Di default coincide con (x, y); la compressione lo standardizza."""
        return self.x, self.y

    # -- suggerimenti opzionali per la visualizzazione -------------------- #
    def hist_range(self) -> tuple[float, float] | None:
        """Intervallo dell'istogramma di fitness (None = auto dai dati)."""
        return None

    def curve_labels(self) -> tuple[str, str]:
        """Etichette (verita', predizione) per il grafico della curva."""
        return ("bersaglio", "predizione del miglior concetto")

    def summarize(self, concept: Concept, lineage: list | None = None) -> dict | None:
        """Traduzione specifica dell'ambiente in linguaggio umano.
        Ritorna {headline, surprise, bullets:[...]} oppure None per usare
        una spiegazione generica. La parte evolutiva (forme, metamorfosi) la
        aggiunge il runner, ed e' indipendente dall'ambiente.
        """
        return None


# --------------------------------------------------------------------------- #
#  Utility condivise dagli ambienti.
# --------------------------------------------------------------------------- #

def shannon_entropy_bits(values: np.ndarray) -> float:
    """Entropia di ordine 0 in bit/simbolo: quanti bit servono in media per
    codificare questi valori con un codificatore entropico ideale."""
    v = np.round(np.asarray(values)).astype(np.int64)
    if v.size == 0:
        return 0.0
    _, counts = np.unique(v, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())
