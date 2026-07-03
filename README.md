# CHIMERA

**C**oncept **H**yperdimensional **I**ntelligence for **M**eta-**E**volutionary
**R**epresentation **A**rchitecture

![license](https://img.shields.io/badge/license-MIT-4dd6c1)
![python](https://img.shields.io/badge/python-3.10%2B-7c9cff)
![deps](https://img.shields.io/badge/deps-numpy%20%C2%B7%20fastapi-8394ab)

> Un motore di **calcolo evolutivo** che fa evolvere soluzioni — funzioni
> matematiche, programmi, permutazioni, stringhe — attraverso mutazione,
> selezione e migrazione, con un osservatorio per guardarlo dal vivo e una sonda
> per misurare la struttura dei problemi.

---

## Cos'è, concretamente

CHIMERA è un **algoritmo evolutivo a isole**, scritto in puro `numpy`. Fa
evolvere una popolazione di *concetti* (soluzioni candidate) con:

- **mutazione**, **accoppiamento** (recombinazione), **selezione** a torneo con elitismo;
- 4 **continenti** con "fisiche" (operatori ammessi) diverse e **migrazione** periodica;
- substrati multipli — un concetto può essere un **programma** (albero di
  espressioni), un **vettore** di feature non lineari o una piccola **rete
  neurale**, e può **cambiare forma** preservando il comportamento (distillazione
  ai minimi quadrati) o **apprendere** i pesi in forma chiusa;
- un **archivio genealogico** completo su SQLite: ogni concetto, i suoi genitori,
  le mutazioni, le prestazioni.

Ci lavori attraverso quattro cose: un **osservatorio web** (dashboard 2D + albero
genealogico 3D) per guardare l'evoluzione dal vivo, un **benchmark scientifico**
(le equazioni di Feynman), una famiglia di **enigmi** combinatori, e una **sonda**
che usa il motore come strumento di misura. Gira su un PC normale.

---

## Cosa fa davvero, oggi (verificato)

- **Riscopre leggi fisiche *note* dai soli dati**, misurate su un train/test
  split — quindi accuratezza su dati mai visti, non memorizzazione. La sua nicchia
  sono le leggi **non-polinomiali** (esponenziali, `1/r²`, gaussiane), dove la
  regressione lineare e polinomiale falliscono (vedi tabella sotto).
- **Risolve enigmi combinatori con lo stesso identico motore** — criptaritmo,
  N regine, commesso viaggiatore — cambiando solo substrato e fitness. Per il TSP
  è incluso il solver esatto (Held-Karp) che **verifica** quanto ci si avvicina
  all'ottimo vero, invece di assumerlo.
- **Mostra, in modo misurabile, quando la ricerca evolutiva funziona**: la scopre
  una frase segreta col feedback graduale, ma è cieca dietro un hash (feedback
  sì/no). È il principio del gradiente, ed è anche il motivo per cui un hash
  protegge una password.
- **Misura la struttura dei problemi** (`chimera/probe.py`): ruggedness,
  gradiente, neutralità, tasso di trappole — verso una tassonomia empirica della
  risolvibilità.

> **Onestà sul benchmark**: sulle equazioni di Feynman CHIMERA *riscopre* leggi
> che **conosciamo già**. È una verifica del metodo su un banco di prova onesto,
> **non** la scoperta di leggi ignote. Quest'ultima è l'obiettivo, non un
> risultato raggiunto.

---

## La scommessa (visione — non ancora realtà)

Oltre a ciò che fa oggi, CHIMERA è la scommessa su un'idea:

> Il *significato* di un concetto non gli è assegnato da noi: **emerge dagli
> effetti** che produce quando interagisce con l'ambiente (la fitness).

Questo è vero *letteralmente* nel codice — un concetto "vale" esattamente quanto
vale il risultato che genera. La scommessa più ambiziosa è che, spingendo questa
regola abbastanza lontano, un sistema del genere possa costruirsi **categorie
proprie** ed esplorare regioni dello spazio delle idee che il pensiero umano non
raggiunge spontaneamente — fino alla **meta-evoluzione**: CHIMERA che riprogetta
i propri mutatori e selettori.

Ad oggi questa parte è **aspirazione, non risultato**. Gli ambienti e i problemi
sono ancora scelti da noi, e il sistema rediscopre soluzioni note. La visione
serve da bussola, non da descrizione di ciò che il codice fa.

---

## Cosa CHIMERA *non* è

- **Non** è un chatbot né un Large Language Model.
- **Non** è un'AGI. È un algoritmo evolutivo con un'ottima infrastruttura di
  osservazione.
- **Non** è un solutore universale: la sonda lo mostra nero su bianco — risolve
  dove il paesaggio offre un gradiente, fallisce dove non c'è (l'hash).
- **Non** ha (ancora) scoperto nulla di ignoto agli umani: sul benchmark
  rediscopre leggi conosciute.
- **Non** scala su GPU né è pensato per la produzione: è un proof-of-concept da
  singolo PC, volutamente leggibile e a zero dipendenze pesanti.

---

## Un motore, molti problemi

Il cuore evolutivo è **uno solo**. Ogni compito cambia **due sole cose**: il
*substrato* (la forma di un concetto) e l'*ambiente* (ciò che ne misura gli
effetti). Il nucleo non cambia mai.

| Problema | Substrato | Ambiente | Comando |
|---|---|---|---|
| Regressione simbolica | funzione (`expression`/`vector`/`mlp`) | `symbolic_regression` | `python -m chimera.serve` |
| Compressione dati | funzione | `compression` | `python -m chimera.serve --env compression` |
| Leggi fisiche (Feynman) | funzione multi-variabile | `feynman:<equazione>` | `python -m chimera.benchmark` |
| Frase segreta | genoma-stringa | segnale graduale / hash | `python -m chimera.enigma` |
| Criptaritmo (SEND+MORE=MONEY) | permutazione di cifre | vincolo aritmetico | `python -m chimera.cryptarithm` |
| N regine | permutazione | conflitti sulle diagonali | `python -m chimera.queens --n 30` |
| Commesso viaggiatore | permutazione | lunghezza del giro | `python -m chimera.tsp` |
| *(misura, non risolve)* | qualsiasi | — | `python -m chimera.probe` |

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
verità-vs-campione, la fitness nel tempo, l'**atlante dei concetti** (mappa 2D
dello spazio dei *comportamenti*), i 4 continenti, la genealogia del campione e
il **traduttore** che spiega il campione in italiano.

### 3) L'albero evolutivo 3D 🌳

Dall'osservatorio, il pulsante **🌳 Albero 3D** (o **/albero**) apre una vista in
cui l'asse verticale è il **tempo**: le idee tracciano traiettorie che salgono,
si biforcano alle nascite, cambiano colore alle metamorfosi e saltano tra i
continenti alle migrazioni.

> *L'albero 3D carica Three.js da CDN: richiede Internet. L'osservatorio 2D
> funziona anche offline.*

---

## Benchmark scientifico — le equazioni di Feynman

Riscopre leggi note dai soli dati, misurato su dati **mai visti** (train/test
split). Baseline = regressione lineare e polinomiale.

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

Due varianti a confronto: **CHIMERA-P** evolve *approssimatori* (ottima
predizione in distribuzione), **CHIMERA-S** evolve *spiegazioni* (programmi che
estrapolano). In estrapolazione, cambiare l'oggetto dell'evoluzione porta da
**1/10 a 7/10** equazioni risolte.

---

## Gli enigmi — la ricerca come discesa lungo un gradiente

L'evoluzione non "indovina": **scala un gradiente**. Funziona se e solo se il
problema offre un segnale graduale.

```bash
# Frase segreta: la scopre col feedback "quante lettere sono giuste"...
python -m chimera.enigma --secret "CHIMERA scopre da sola"
# ...ma dietro un hash (feedback solo sì/no) è cieca — ecco perché un hash
# protegge una password:
python -m chimera.enigma --secret "CHIMERA scopre da sola" --mode hash

# Criptaritmo: risposta unica imposta dall'aritmetica, non da noi
python -m chimera.cryptarithm                       # 9567 + 1085 = 10652

# N regine: vincoli geometrici, risposta verificabile
python -m chimera.queens --n 30

# Commesso viaggiatore su città vere + solver esatto (Held-Karp) per verifica
python -m chimera.tsp
```

---

## La sonda — misurare la struttura di un problema

Rovescia la domanda: non "il motore risolve?" ma "che forma ha il paesaggio del
problema?". Usa il motore come **strumento di misura**, non come solutore.

```bash
python -m chimera.probe
```

Per ogni problema stima, con lo stesso encoding a confronto: **ruggedness**
(quanto è accidentato il terreno, normalizzata sul diametro), **densità di
gradiente** (quanto spesso si può salire), **neutralità** (plateau senza
pendenza) e **tasso di trappole** (frazione di salite che si bloccano su un
ottimo locale non-soluzione — senza bisogno di conoscere l'ottimo). Ne emerge,
tra l'altro, che frase e hash hanno terreni quasi ugualmente piatti, ma la frase
ha un filo di gradiente e zero trappole (risolvibile) mentre l'hash no
(impossibile) — la differenza tra difficile e impossibile, in numeri.

> La sonda misura la difficoltà **per questa dinamica fissa**: è un'impronta del
> paesaggio, non una verità assoluta. Cambiando l'encoding, il terreno cambia.

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

Non salviamo *il modello*: salviamo **l'albero dell'evoluzione**. Ogni concetto,
i suoi genitori, le sue prestazioni. È l'oggetto che l'albero 3D rende visibile.

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
├── probe.py             # sonda: misura la struttura dei problemi
├── main.py              # motore da CLI
├── serve.py             # avvia l'osservatorio
└── benchmark.py         # benchmark Feynman
```

---

## Roadmap

- **v0.1–0.6** ✓ — nucleo evolutivo, osservatorio, substrati multipli e
  metamorfosi, compressione, benchmark Feynman, CHIMERA-S (evolvere leggi).
- **osservatorio 3D** ✓ — l'albero genealogico nel tempo; ambienti Feynman
  nella dashboard.
- **enigmi combinatori** ✓ — stesso motore su frase, criptaritmo, regine, TSP.
- **sonda dei problemi** ✓ — il motore come strumento di misura del paesaggio.
- **prossimo** — allargare la mappa (più problemi ed encoding) e verificare se la
  struttura misurata *predice* la velocità di convergenza.
- **visione** — meta-evoluzione: CHIMERA che riprogetta i propri mutatori,
  selettori e traduttori. *(Non ancora iniziata.)*

---

## Contribuire

Il progetto è volutamente **a zero dipendenze pesanti** e leggibile. Il modo più
naturale di contribuire è aggiungere un **nuovo ambiente** o un **nuovo
substrato** senza toccare il nucleo:

- un ambiente implementa `evaluate(concept) -> (fitness, beats_sota)` e
  `sota_score()` (vedi `environments/base.py`);
- un substrato implementa `mutate` / `recombine` / `describe` / `complexity` /
  `kind` (vedi `core/substrate.py` e `combinatorial.py`).

Issue e pull request benvenute.

---

## Licenza

[MIT](LICENSE) © 2026 Alessandro Di Vito
