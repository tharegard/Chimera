# CHIMERA

**C**oncept **H**yperdimensional **I**ntelligence for **M**eta-**E**volutionary
**R**epresentation **A**rchitecture

![license](https://img.shields.io/badge/license-MIT-4dd6c1)
![python](https://img.shields.io/badge/python-3.10%2B-7c9cff)
![deps](https://img.shields.io/badge/deps-numpy%20%C2%B7%20fastapi-8394ab)

> Un motore per l'**evoluzione autonoma dei concetti**: fa evolvere idee
> matematiche come fossero organismi, e osserva la loro genealogia crescere.

---

## Visione

CHIMERA **non** è un chatbot. **Non** è un Large Language Model. **Non** è
un'AGI nel senso tradizionale.

CHIMERA è una macchina che **esplora lo spazio delle rappresentazioni**.
L'obiettivo non è rispondere meglio alle domande umane, ma esplorare regioni
dello spazio delle idee che il pensiero umano potrebbe non raggiungere
spontaneamente.

### La regola che non cambieremo mai

> In CHIMERA **nessun concetto ha un significato assegnato dagli esseri umani**.
> Il significato emerge esclusivamente dagli effetti che il concetto produce
> quando interagisce con altri concetti e con l'ambiente.

Questa regola costringe il sistema a costruire le proprie categorie invece di
ereditare le nostre.

### Principi fondamentali

1. Nessun linguaggio umano come rappresentazione interna.
2. I concetti sono oggetti matematici, non parole.
3. Ogni concetto può mutare, fondersi, dividersi o scomparire.
4. Solo i risultati determinano la sopravvivenza.
5. Il sistema evolve continuamente la propria rappresentazione della conoscenza.
6. L'essere umano osserva l'evoluzione, ma non la dirige nel dettaglio.

---

## Un motore, molti problemi

Il cuore di CHIMERA — popolazione, mutazione, accoppiamento, selezione,
continenti, migrazione, archivio genealogico — è **uno solo**. Ogni nuovo
compito si aggiunge cambiando **due sole cose**: il *substrato* (la forma
concreta di un concetto) e l'*ambiente* (ciò che ne misura gli effetti). Il
nucleo evolutivo non cambia mai. È la dimostrazione concreta della tesi del
progetto: *ogni parte del sistema può essere sostituita*.

| Problema | Substrato | Ambiente | Comando |
|---|---|---|---|
| Regressione simbolica | funzione (`expression`/`vector`/`mlp`) | `symbolic_regression` | `python -m chimera.serve` |
| Compressione dati | funzione | `compression` | `python -m chimera.serve --env compression` |
| Leggi fisiche (Feynman) | funzione multi-variabile | `feynman:<equazione>` | `python -m chimera.benchmark` |
| Frase segreta | genoma-stringa | segnale graduale / hash | `python -m chimera.enigma` |
| Criptaritmo (SEND+MORE=MONEY) | permutazione di cifre | vincolo aritmetico | `python -m chimera.cryptarithm` |
| N regine | permutazione | conflitti sulle diagonali | `python -m chimera.queens --n 30` |
| Commesso viaggiatore | permutazione | lunghezza del giro | `python -m chimera.tsp` |

---

## Installazione

Serve **Python 3.10+**. Solo `numpy` per il motore; `fastapi`/`uvicorn` per
l'osservatorio web.

```bash
git clone https://github.com/tharegard/Chimera.git
cd Chimera
pip install -r requirements.txt
```

---

## Avvio rapido

### 1) Il motore da riga di comando

```bash
python -m chimera.main --generations 60 --pop 120
```

Stampa la genealogia completa: nascite, campioni, migrazioni.

### 2) L'osservatorio web — guardare l'evoluzione dal vivo

```bash
python -m chimera.serve                      # regressione simbolica
python -m chimera.serve --env feynman:gaussiana
```

Apri **http://127.0.0.1:8000**. Premi *Avvia* e osserva: la curva
verità-vs-campione, la fitness nel tempo, l'**atlante dei concetti** (una mappa
2D dello spazio dei *comportamenti*), i 4 continenti, la genealogia del campione
e il **traduttore** che spiega il campione in italiano.

### 3) L'albero evolutivo 3D 🌳

Dall'osservatorio, il pulsante **🌳 Albero 3D** (o **http://127.0.0.1:8000/albero**)
apre una vista tridimensionale in cui l'asse verticale è il **tempo**: le idee
tracciano traiettorie che salgono, si biforcano alle nascite, cambiano colore
alle metamorfosi e saltano tra i continenti alle migrazioni. Non uno snapshot:
la specie che cresce.

> *L'albero 3D carica la libreria Three.js da CDN: richiede una connessione a
> Internet. L'osservatorio 2D funziona anche offline.*

---

## Benchmark scientifico — le equazioni di Feynman

CHIMERA riscopre leggi fisiche note da soli dati, misurato su un **train/test
split** (errore su dati mai visti — non memorizzazione). Baseline = regressione
lineare e polinomiale.

```bash
python -m chimera.benchmark --generations 80 --pop 120
```

| equazione | formula | R² CHIMERA | R² lin | R² poly | esito |
|---|---|---|---|---|---|
| prodotto | a·b | 1.000 | 0.928 | 1.000 | risolto |
| cinetica | ½·m·v² | 1.000 | 0.877 | 0.997 | risolto |
| gravitazione | m1·m2/r² | 0.983 | 0.613 | 0.888 | batte baseline |
| gaussiana | e^(−t²/2)/√(2π) | 1.000 | −0.007 | 0.809 | risolto |
| pendolo | 2π·√(L/g) | 1.000 | 0.990 | 1.000 | risolto |
| smorzamento | A·e^(−x) | 1.000 | 0.809 | 0.981 | risolto |

**La nicchia di CHIMERA sono le leggi non-polinomiali** (exp, 1/r², gaussiana,
divisione), dove la regressione polinomiale fallisce. Due varianti a confronto:
**CHIMERA-P** evolve *approssimatori* (ottima predizione in distribuzione),
**CHIMERA-S** evolve *spiegazioni* (programmi che estrapolano): in
estrapolazione passa da **1/10 a 7/10** equazioni risolte.

---

## Gli enigmi — la ricerca come discesa lungo un gradiente

Una famiglia di dimostrazioni che chiariscono *quando* la ricerca evolutiva
funziona. Il principio: l'evoluzione non "indovina", **scala un gradiente**.
Funziona se e solo se il problema offre un segnale graduale.

```bash
# Frase segreta: la scopre col feedback "quante lettere sono giuste"...
python -m chimera.enigma --secret "CHIMERA scopre da sola"
# ...ma dietro un hash (feedback solo sì/no) l'evoluzione è cieca — ecco
# perché un hash protegge una password:
python -m chimera.enigma --secret "CHIMERA scopre da sola" --mode hash

# Criptaritmo: risposta unica imposta dall'aritmetica, non da noi
python -m chimera.cryptarithm                       # 9567 + 1085 = 10652
python -m chimera.cryptarithm --addends TWO TWO --result FOUR

# N regine: vincoli geometrici, risposta verificabile
python -m chimera.queens --n 30

# Commesso viaggiatore su città italiane vere: nessuno conosce l'ottimo.
# Include il solver esatto (Held-Karp) per verificare quanto ci si avvicina.
python -m chimera.tsp
```

Questi enigmi non sono l'obiettivo finale: sono un **banco di prova** per capire
come cambia la difficoltà di un problema al variare di *encoding + fitness*.

---

## Architettura

```
Problema
   │
Popolazione di concetti  ──► Continente A · B · C · D  (matematiche/fisiche diverse)
   │                              │
Mutazione + Accoppiamento         │  Migrazione periodica
   │                              │
Ambiente di simulazione ◄─────────┘
   │
Fitness  (batte lo stato dell'arte?  vive : muore)
   │
Archivio evolutivo   (albero genealogico completo — NON un file .pt)
```

Non salviamo *il modello*. Salviamo **l'albero dell'evoluzione**: ogni concetto,
i suoi genitori, le mutazioni, le prestazioni. CHIMERA non è un modello, è una
**specie digitale**: può migrare, adattarsi, continuare a evolversi anche se
cambia l'hardware.

### Mappa del repository

```
chimera/
├── core/
│   ├── concept.py       # l'unità di evoluzione: identità, genitori, fitness
│   ├── substrate.py     # le forme: expression · vector · mlp · metamorfosi
│   ├── evolution.py     # World / Continent: selezione, riproduzione, migrazione
│   └── archive.py       # l'albero genealogico su SQLite
├── environments/
│   ├── base.py          # il contratto: evaluate() + sota_score()
│   ├── symbolic_regression.py
│   ├── compression.py
│   └── feynman.py       # le equazioni fisiche (train/test)
├── ui/
│   ├── server.py        # API FastAPI dell'osservatorio
│   ├── runner.py        # il motore in un thread di background
│   ├── index.html       # dashboard 2D
│   └── tree.html        # albero evolutivo 3D (Three.js)
├── combinatorial.py     # substrato-permutazione condiviso (regine, TSP)
├── enigma.py            # frase segreta (graduale vs hash)
├── cryptarithm.py       # SEND + MORE = MONEY
├── queens.py            # N regine
├── tsp.py               # commesso viaggiatore + solver esatto Held-Karp
├── main.py              # motore da CLI
├── serve.py             # avvia l'osservatorio
├── benchmark.py         # benchmark Feynman
└── probe.py             # sonda: misura la struttura dei problemi
```

---

## Roadmap

- **v0.1–0.6** ✓ — nucleo evolutivo, osservatorio, substrati multipli e
  metamorfosi, compressione, benchmark Feynman, CHIMERA-S (evolvere leggi).
- **osservatorio 3D** ✓ — l'albero genealogico nel tempo; ambienti Feynman
  agganciati alla dashboard.
- **enigmi combinatori** ✓ — stesso motore su frase, criptaritmo, regine, TSP.
- **sonda dei problemi** ✓ — `python -m chimera.probe`: usa il motore come
  *strumento di misura* del paesaggio di un problema (ruggedness normalizzata,
  densità di gradiente, neutralità, tasso di trappole), verso una **tassonomia
  empirica della risolvibilità**. Non "risolve", *misura*.
- **prossimo** — allargare la mappa (più problemi ed encoding), normalizzare i
  confronti cross-encoding, e mettere in relazione la struttura misurata con la
  velocità di convergenza del motore.
- **visione** — meta-evoluzione: CHIMERA che riprogetta i propri mutatori,
  selettori e traduttori — evolve il proprio processo evolutivo.

> Ogni parte del sistema deve poter essere sostituita dall'evoluzione stessa.

---

## Contribuire

Il progetto è volutamente **a zero dipendenze pesanti** e leggibile. Il modo più
naturale di contribuire è aggiungere un **nuovo ambiente** o un **nuovo
substrato** senza toccare il nucleo:

- un ambiente implementa `evaluate(concept) -> (fitness, beats_sota)` e
  `sota_score()` (vedi `environments/base.py`);
- un substrato implementa `mutate` / `recombine` / `describe` / `complexity` /
  `kind` (vedi `core/substrate.py` e `combinatorial.py`).

Apri pure issue e pull request.

---

## Licenza

[MIT](LICENSE) © 2026 Alessandro Di Vito
