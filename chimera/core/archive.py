"""
archive.py
==========

L'Archivio Evolutivo e' il vero prodotto di CHIMERA.

Non salviamo "il modello". Salviamo l'ALBERO dell'evoluzione: ogni concetto
mai nato, i suoi genitori, la mutazione che l'ha generato, la sua prestazione.
Cosi' possiamo tornare indietro nel tempo e capire *perche'* un concetto e'
emerso e chi erano i suoi antenati. E' una specie digitale, non un file .pt.

Backend: SQLite (zero dipendenze, gia' nella stdlib di Python).
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from .concept import Concept

_SCHEMA = """
CREATE TABLE IF NOT EXISTS concepts (
    id          INTEGER PRIMARY KEY,
    continent   TEXT,
    generation  INTEGER,
    origin      TEXT,
    parents     TEXT,          -- csv di id
    fitness     REAL,
    beats_sota  INTEGER,
    complexity  INTEGER,
    structure   TEXT,          -- descrizione leggibile (solo per l'umano)
    born_run    TEXT,          -- identificativo della run
    kind        TEXT           -- forma del substrato (expression|vector|mlp)
);
CREATE INDEX IF NOT EXISTS idx_continent ON concepts(continent);
CREATE INDEX IF NOT EXISTS idx_fitness   ON concepts(fitness);
CREATE INDEX IF NOT EXISTS idx_parents   ON concepts(parents);
"""


class Archive:
    def __init__(self, path: str | Path = "chimera_archive.sqlite", run_id: str = "run"):
        self.path = str(path)
        self.run_id = run_id
        # check_same_thread=False + lock: il motore scrive da un thread,
        # l'API di visualizzazione legge da un altro (vedi ui/server.py).
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        self.conn.executescript(_SCHEMA)
        # migrazione dolce per archivi creati prima della v0.3
        try:
            self.conn.execute("ALTER TABLE concepts ADD COLUMN kind TEXT")
        except sqlite3.OperationalError:
            pass
        self.conn.commit()

    def _insert(self, concept: Concept) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO concepts "
            "(id, continent, generation, origin, parents, fitness, beats_sota, "
            " complexity, structure, born_run, kind) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                concept.id,
                concept.continent,
                concept.generation,
                concept.origin,
                ",".join(str(p) for p in concept.parents),
                concept.fitness,
                int(concept.beats_sota),
                concept.substrate.complexity(),
                concept.describe(),
                self.run_id,
                concept.substrate.kind,
            ),
        )

    def record(self, concept: Concept) -> None:
        with self._lock:
            self._insert(concept)
            self.conn.commit()

    def record_many(self, concepts) -> None:
        with self._lock:
            for c in concepts:
                self._insert(c)
            self.conn.commit()

    # -- interrogazioni ---------------------------------------------------- #
    def best(self, limit: int = 1):
        with self._lock:
            cur = self.conn.execute(
                "SELECT id, continent, generation, fitness, beats_sota, complexity, structure, kind "
                "FROM concepts ORDER BY fitness DESC LIMIT ?",
                (limit,),
            )
            return cur.fetchall()

    def lineage(self, concept_id: int) -> list[tuple]:
        """Risale l'albero genealogico fino ai capostipiti (genitore per genitore)."""
        chain = []
        frontier = [concept_id]
        seen = set()
        with self._lock:
            while frontier:
                cid = frontier.pop()
                if cid in seen:
                    continue
                seen.add(cid)
                row = self.conn.execute(
                    "SELECT id, origin, parents, fitness, structure FROM concepts WHERE id=?",
                    (cid,),
                ).fetchone()
                if not row:
                    continue
                chain.append(row)
                parents = [int(p) for p in row[2].split(",") if p]
                frontier.extend(parents)
        return sorted(chain, key=lambda r: r[0])

    def stats(self) -> dict:
        with self._lock:
            total = self.conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
            winners = self.conn.execute(
                "SELECT COUNT(*) FROM concepts WHERE beats_sota=1"
            ).fetchone()[0]
            per_cont = dict(
                self.conn.execute(
                    "SELECT continent, COUNT(*) FROM concepts GROUP BY continent"
                ).fetchall()
            )
        return {"total": total, "beat_sota": winners, "per_continent": per_cont}

    def clear(self) -> None:
        """Svuota l'archivio (usato dal reset della visualizzazione)."""
        with self._lock:
            self.conn.execute("DELETE FROM concepts")
            self.conn.commit()

    def close(self) -> None:
        with self._lock:
            self.conn.commit()
            self.conn.close()
