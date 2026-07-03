"""
runner.py
=========

Orchestratore del motore per la visualizzazione.

Fa girare il World in un thread di background (start/pausa/reset) e, dopo ogni
generazione, pubblica uno "stato" in un dizionario semplice che l'API puo'
leggere senza toccare gli oggetti vivi (niente race condition).

REGOLA v0.2: il browser OSSERVA. Tutta l'evoluzione avviene qui, in Python.
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque

import numpy as np

from ..core.archive import Archive
from ..core.evolution import Continent, World
from ..core.substrate import ExpressionTree, Physics, TinyMLP, VectorFeature
from ..environments.compression import PredictiveCompression
from ..environments.feynman import EQUATIONS, FeynmanEnvironment
from ..environments.symbolic_regression import SymbolicRegression

ENVIRONMENTS = {
    "symbolic_regression": lambda seed: SymbolicRegression(seed=seed),
    "compression": lambda seed: PredictiveCompression(),
    # benchmark scientifico: "feynman" = gaussiana (monovariabile, mostrabile
    # anche nella curva 2D); ogni equazione anche come "feynman:<nome>".
    "feynman": lambda seed: FeynmanEnvironment("gaussiana", seed=seed),
}
for _key in EQUATIONS:
    ENVIRONMENTS[f"feynman:{_key}"] = (
        lambda k: (lambda seed: FeynmanEnvironment(k, seed=seed))
    )(_key)


def make_environment(name: str, seed: int):
    if name not in ENVIRONMENTS:
        raise ValueError(f"ambiente sconosciuto: {name}. Disponibili: {list(ENVIRONMENTS)}")
    return ENVIRONMENTS[name](seed)

# Fisiche dei continenti: ognuno una "matematica" diversa (operatori ammessi).
CONTINENT_SPECS = [
    ("Alpha", ("neg", "abs", "sin", "cos"),          ("add", "sub", "mul")),
    ("Beta",  ("tanh", "square", "neg"),             ("add", "mul", "div")),
    ("Gamma", ("log", "sqrt", "abs", "square"),      ("add", "sub", "mul", "div")),
    ("Delta", ("sin", "cos", "tanh", "neg", "abs"),  ("add", "sub", "mul", "div")),
]

HIST_LO, HIST_HI, HIST_BINS = -1.5, 0.0, 14
CURVE_POINTS = 70
BEHAVIOR_DIM = 20        # punti su cui si misura il "comportamento" di un concetto
ATLAS_SAMPLE = 55        # concetti campionati per continente nell'atlante
TREE_TOPK = 12           # nodi (per continente/generazione) aggiunti all'albero 3D


class EngineRunner:
    def __init__(self, pop=120, seed=0, delay=0.25, max_generations=400,
                 archive_path="chimera_archive.sqlite", env_name="symbolic_regression"):
        self.pop = pop
        self.seed = seed
        self.delay = delay
        self.max_generations = max_generations
        self.archive_path = archive_path
        self.env_name = env_name

        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self.events: deque = deque(maxlen=60)
        self._best_history: list[float] = []
        self._cont_history: dict[str, list[float]] = {}
        self._global_best = float("-inf")
        self.state: dict = {}

        self._build_world()

    # -- costruzione / reset ---------------------------------------------- #
    def _build_world(self):
        self.env = make_environment(self.env_name, self.seed)
        self.archive = Archive(self.archive_path, run_id=f"run-{self.seed}")
        rng = lambda s: __import__("random").Random(self.seed + s)
        n_vars = int(getattr(self.env, "n_vars", 1))   # ambienti multivariabile (Feynman)
        continents = []
        for i, (name, unary, binary) in enumerate(CONTINENT_SPECS):
            physics = Physics(unary=unary, binary=binary, max_depth=4,
                              n_vars=n_vars, rng=rng(i + 1))
            population = World.seed_population(physics, name, self.pop)
            continents.append(Continent(name=name, physics=physics, population=population))
        self.world = World(self.env, continents, self.archive, seed=self.seed)
        self.world.bootstrap()
        self._best_history = []
        self._cont_history = {c.name: [] for c in continents}
        self._global_best = float("-inf")
        self._champion = None
        self.events.clear()
        self._atlas_setup()
        self._tree_reset()
        self._publish(migrated=False)
        self._log("genesi", f"{len(continents)} continenti, {self.pop} concetti ciascuno")

    # -- atlante: proiezione 2D dello spazio dei comportamenti ------------- #
    def _atlas_setup(self):
        """Congela una proiezione (PCA) una volta sola: la mappa resta stabile
        mentre la popolazione si muove. Gli assi sono definiti dallo spazio dei
        comportamenti, non dalla struttura -> guardiamo cosa fanno i concetti."""
        self._beh_idx = np.linspace(0, len(self.env.x) - 1, BEHAVIOR_DIM).astype(int)
        lo, hi = float(self.env.y.min()), float(self.env.y.max())
        pad = 0.6 * (hi - lo + 1e-6)
        self._beh_clip = (lo - pad, hi + pad)

        phys = self.world.continents[0].physics
        ref = []
        for _ in range(120):
            r = random.random()
            if r < 0.5:
                sub = ExpressionTree.random(phys)
            elif r < 0.75:
                sub = VectorFeature.random(phys)
            else:
                sub = TinyMLP.random(phys)
            ref.append(self._behavior(sub))
        ref.append(self._behavior_of_target())
        B = np.array(ref)
        self._atlas_mean = B.mean(axis=0)
        _, _, Vt = np.linalg.svd(B - self._atlas_mean, full_matrices=False)
        self._atlas_comps = Vt[:2].T                      # (BEHAVIOR_DIM, 2)
        P = (B - self._atlas_mean) @ self._atlas_comps
        self._atlas_bounds = [float(P[:, 0].min()), float(P[:, 0].max()),
                              float(P[:, 1].min()), float(P[:, 1].max())]

    def _behavior(self, substrate) -> np.ndarray:
        b = self.env.predict(substrate, self.env.x[self._beh_idx])
        return np.clip(np.nan_to_num(b), *self._beh_clip)

    def _behavior_of_target(self) -> np.ndarray:
        return np.clip(self.env.y[self._beh_idx], *self._beh_clip)

    def _project(self, beh: np.ndarray):
        p = (beh - self._atlas_mean) @ self._atlas_comps
        return [round(float(p[0]), 3), round(float(p[1]), 3)]

    def _atlas_points(self):
        pts = []
        for c in self.world.continents:
            sample = c.population if len(c.population) <= ATLAS_SAMPLE else \
                random.sample(c.population, ATLAS_SAMPLE)
            for k in sample:
                xy = self._project(self._behavior(k.substrate))
                pts.append({"x": xy[0], "y": xy[1], "form": k.substrate_kind,
                            "cont": c.name, "win": bool(k.beats_sota)})
        return pts

    # -- albero evolutivo 3D: lo spazio dei comportamenti nel TEMPO -------- #
    # Non uno snapshot: la specie che cresce. Piano X-Y = comportamenti (le
    # stesse 2 PCA congelate dell'atlante), asse "tempo" = generazione. Ogni
    # concetto e' un nodo alla propria altezza; gli archi sono la genealogia
    # REALE (parents): biforcazione alla nascita, fusione all'accoppiamento,
    # salto alla migrazione. E' l'oggetto che CHIMERA dichiara di essere.
    def _tree_reset(self):
        self._tree_nodes: list[dict] = []     # append-only
        self._tree_index: dict[int, int] = {}  # id concetto -> posizione nel nodo
        self._tree_edges: list[list] = []      # [pos_antenato, pos_nodo, kind]
        self._parent_of: dict[int, tuple] = {}  # id -> genitori (solo interi: costa poco)

    def _find_plotted_ancestor(self, parent_id: int):
        """Risale la catena genealogica reale fino al primo antenato che e'
        effettivamente disegnato: cosi' l'albero resta connesso anche se i
        concetti intermedi non sono stati plottati (potatura filogenetica)."""
        frontier = [parent_id]
        seen: set[int] = set()
        while frontier:
            p = frontier.pop()
            if p in self._tree_index:
                return p
            if p in seen:
                continue
            seen.add(p)
            frontier.extend(self._parent_of.get(p, ()))
        return None

    def _tree_capture(self):
        g = self.world.generation
        # 1) memorizza i genitori di TUTTI i viventi (solo interi)
        for c in self.world.continents:
            for k in c.population:
                if k.id not in self._parent_of:
                    self._parent_of[k.id] = tuple(k.parents)
        # 2) nuovi nodi = i concetti piu' forti mai visti, per continente
        for c in self.world.continents:
            newcomers = [k for k in c.population if k.id not in self._tree_index]
            newcomers.sort(key=lambda k: k.fitness, reverse=True)
            for k in newcomers[:TREE_TOPK]:
                xy = self._project(self._behavior(k.substrate))
                pos = len(self._tree_nodes)
                self._tree_index[k.id] = pos
                self._tree_nodes.append({
                    "id": k.id, "x": xy[0], "y": xy[1], "gen": g,
                    "form": k.substrate_kind, "cont": k.continent,
                    "win": bool(k.beats_sota), "fit": round(float(k.fitness), 4),
                    "origin": k.origin,
                })
                # 3) rami reali verso il primo antenato plottato
                linked: set[int] = set()
                for par in (k.parents or ()):
                    anc = self._find_plotted_ancestor(par)
                    if anc is not None and anc not in linked:
                        linked.add(anc)
                        kind = "migration" if k.origin == "migration" else \
                               "recombination" if k.origin == "recombination" else \
                               "metamorphosis" if k.origin == "metamorphosis" else "descent"
                        self._tree_edges.append([self._tree_index[anc], pos, kind])

    def _tree_champion_path(self) -> list[int]:
        """Posizioni (nel vettore dei nodi) della stirpe principale del campione,
        risalendo il primo genitore: la spina dorsale da evidenziare in oro."""
        champ = getattr(self, "_champion", None)
        if champ is None:
            return []
        path, cur, seen = [], champ.id, set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            pos = self._tree_index.get(cur)
            if pos is not None:
                path.append(pos)
            pars = self._parent_of.get(cur, ())
            cur = pars[0] if pars else None
        return path

    def get_tree(self, since_nodes: int = 0, since_edges: int = 0) -> dict:
        try:
            total_n, total_e = len(self._tree_nodes), len(self._tree_edges)
            if since_nodes > total_n or since_edges > total_e:  # dopo un reset
                since_nodes = since_edges = 0
            return {
                "nodes": self._tree_nodes[since_nodes:],
                "edges": self._tree_edges[since_edges:],
                "total_nodes": total_n,
                "total_edges": total_e,
                "bounds": self._atlas_bounds,
                "target": self._project(self._behavior_of_target()),
                "generation": self.world.generation,
                "max_generations": self.max_generations,
                "running": self._running,
                "champion_id": self._champion.id if getattr(self, "_champion", None) else None,
                "champion_path": self._tree_champion_path(),
            }
        except Exception:  # lettura concorrente durante la pubblicazione: riprova al prossimo poll
            return {"nodes": [], "edges": [], "total_nodes": since_nodes,
                    "total_edges": since_edges, "generation": self.world.generation}

    # -- ciclo di background ---------------------------------------------- #
    def _loop(self):
        while self._running and self.world.generation < self.max_generations:
            migrated = (self.world.generation + 1) % self.world.migration_every == 0
            self.world.step()
            self._publish(migrated=migrated)
            time.sleep(self.delay)
        self._running = False

    def start(self):
        with self._lock:
            if self._running:
                return
            if self.world.generation >= self.max_generations:
                return
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def pause(self):
        self._running = False

    def reset(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.archive.clear()
        self._build_world()

    def set_speed(self, delay: float):
        self.delay = max(0.0, min(2.0, float(delay)))

    # -- pubblicazione dello stato ---------------------------------------- #
    def _publish(self, migrated: bool):
        conts = []
        living_best = None
        forms_total = {"expression": 0, "vector": 0, "mlp": 0}
        for c in self.world.continents:
            fits = [k.fitness for k in c.population if np.isfinite(k.fitness)]
            best = max(c.population, key=lambda k: k.fitness)
            if living_best is None or best.fitness > living_best.fitness:
                living_best = best
            self._cont_history[c.name].append(best.fitness)
            forms = {"expression": 0, "vector": 0, "mlp": 0}
            for k in c.population:
                forms[k.substrate_kind] = forms.get(k.substrate_kind, 0) + 1
                forms_total[k.substrate_kind] = forms_total.get(k.substrate_kind, 0) + 1
            conts.append({
                "name": c.name,
                "size": len(c.population),
                "best_fitness": round(best.fitness, 4),
                "beats_sota": sum(1 for k in c.population if k.beats_sota),
                "histogram": self._histogram(fits),
                "history": [round(v, 4) for v in self._cont_history[c.name][-80:]],
                "forms": forms,
            })

        self._best_history.append(living_best.fitness)

        # eventi: nuovo campione globale / migrazione
        if living_best.fitness > self._global_best + 1e-6:
            self._global_best = living_best.fitness
            beat = "  BATTE lo stato dell'arte" if living_best.beats_sota else ""
            self._log("campione", f"gen {self.world.generation} - fitness {living_best.fitness:.4f}{beat}", cont=living_best.continent)
        if migrated:
            self._log("migrazione", f"gen {self.world.generation} - i migliori concetti attraversano i continenti")
        n_meta = self.world.last_births.get("metamorphosis", 0)
        if n_meta:
            self._log("metamorfosi", f"gen {self.world.generation} - {n_meta} concetti hanno cambiato forma (programma → vettore/rete)")
        n_learn = self.world.last_births.get("learning", 0)
        if n_learn:
            self._log("apprendimento", f"gen {self.world.generation} - {n_learn} concetti hanno adattato i pesi all'ambiente")

        self._champion = living_best
        pred = self.env.predict(living_best.substrate, self.env.x)
        X = self.env.x
        if X.ndim == 1:
            idx = np.linspace(0, len(X) - 1, CURVE_POINTS).astype(int)
            xcoord = X[idx]
        else:
            # multivariabile: nessuna ascissa naturale. Ordiniamo per bersaglio
            # e usiamo la posizione normalizzata: la curva mostra se il concetto
            # segue la legge vera lungo tutto il campione (verita' monotona).
            order = np.argsort(self.env.y)
            idx = order[np.linspace(0, len(order) - 1, CURVE_POINTS).astype(int)]
            xcoord = np.linspace(0.0, 1.0, CURVE_POINTS)

        stats = self.archive.stats()
        atlas = {
            "points": self._atlas_points(),
            "target": self._project(self._behavior_of_target()),
            "champion": self._project(self._behavior(living_best.substrate)),
            "bounds": self._atlas_bounds,
        }

        self.state = {
            "generation": self.world.generation,
            "max_generations": self.max_generations,
            "running": self._running,
            "delay": self.delay,
            "environment": self.env.name,
            "curve_labels": list(self.env.curve_labels()),
            "sota_fitness": round(self.env.sota_score(), 4),
            "global_best": round(self._global_best, 4),
            "best_history": [round(v, 4) for v in self._best_history[-120:]],
            "continents": conts,
            "champion": {
                "id": living_best.id,
                "continent": living_best.continent,
                "generation": living_best.generation,
                "fitness": round(living_best.fitness, 4),
                "beats_sota": bool(living_best.beats_sota),
                "complexity": living_best.substrate.complexity(),
                "structure": living_best.describe(),
                "form": living_best.substrate_kind,
            },
            "forms_total": forms_total,
            "curve": {
                "x": [round(float(v), 3) for v in xcoord],
                "target": [round(float(v), 3) for v in self.env.y[idx]],
                "pred": [round(float(v), 3) for v in pred[idx]],
            },
            "archive": stats,
            "atlas": atlas,
            "events": list(self.events),
        }
        self._tree_capture()

    def _histogram(self, fits):
        if not fits:
            return [0] * HIST_BINS
        rng = self.env.hist_range()
        if rng is None:
            lo, hi = float(np.min(fits)), float(np.max(fits))
            if hi - lo < 1e-6:
                hi = lo + 1.0
        else:
            lo, hi = rng
        arr = np.clip(np.array(fits), lo, hi)
        counts, _ = np.histogram(arr, bins=HIST_BINS, range=(lo, hi))
        return [int(v) for v in counts]

    def _log(self, kind: str, text: str, cont: str = ""):
        self.events.appendleft({
            "kind": kind,
            "text": text,
            "continent": cont,
            "generation": self.world.generation,
        })

    # -- accesso -------------------------------------------------------------
    def get_state(self) -> dict:
        return self.state

    def get_lineage(self, concept_id: int) -> list[dict]:
        rows = self.archive.lineage(concept_id)
        return [
            {"id": r[0], "origin": r[1], "parents": r[2], "fitness": round(r[3], 4), "structure": r[4]}
            for r in rows
        ]

    # -- traduttore: dal concetto al linguaggio umano --------------------- #
    def get_translation(self) -> dict:
        """Spiega il campione in termini umani. La parte SPECIFICA dell'ambiente
        (cosa fa, quanto e' bravo) la fornisce l'ambiente via summarize(); la
        parte EVOLUTIVA (forma, stirpe, metamorfosi) e' generica e la aggiunge
        qui il runner. E' l'ultima casella dell'architettura: il traduttore."""
        champ = self._champion
        if champ is None:
            return {"headline": "Nessun campione ancora.", "bullets": []}

        lin = self.archive.lineage(champ.id)
        domain = self.env.summarize(champ, lin) or {
            "headline": f"Campione con fitness {champ.fitness:.4f}.",
            "surprise": "",
            "bullets": [f"Struttura: {champ.describe()}"],
        }

        form_it = {"expression": "un programma", "vector": "un vettore",
                   "mlp": "una rete neurale"}[champ.substrate_kind]
        bullets = list(domain.get("bullets", []))

        # parte evolutiva generica (indipendente dall'ambiente)
        n_meta = sum(1 for r in lin if r[1] == "metamorphosis")
        forms_seen = {self._kind_of(r[4]) for r in lin}
        if n_meta > 0 or len(forms_seen) > 1:
            viaggio = " → ".join({"expression": "programma", "vector": "vettore", "mlp": "rete"}[f]
                                 for f in ["expression", "vector", "mlp"] if f in forms_seen)
            bullets.append(f"Il suo comportamento nasce da una stirpe di {len(lin)} antenati e "
                           f"{n_meta} metamorfosi (forme attraversate: {viaggio}).")

        return {
            "headline": f"Il campione è {form_it}. " + domain.get("headline", ""),
            "form": champ.substrate_kind,
            "surprise": domain.get("surprise", ""),
            "bullets": bullets,
        }

    @staticmethod
    def _kind_of(structure: str) -> str:
        if structure.startswith("mlp("):
            return "mlp"
        if structure.startswith("vector["):
            return "vector"
        return "expression"
