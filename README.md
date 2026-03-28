# Audience Digital Twin — COM-B + ELM ABM

An agent-based digital twin for simulating audience response to persuasive messaging.  
Integrates the **COM-B** behavioural feasibility framework with the **Elaboration Likelihood Model (ELM)** for route-dependent persuasion and attitude durability.

> Companion code for: *"A Verifiable Agent-Based Digital Twin Architecture for Audience Response: Integrating COM-B and ELM for Durable Behaviour Simulation"* — ICSOFT 2026 Position Paper.

---

## What it does

Each agent maintains explicit state variables for capability, opportunity, reflective and automatic motivation, cognitive load, attitude direction, and attitude strength. At every timestep, agents are exposed to messages (via peer diffusion and an engagement-based visibility proxy), process them via central or peripheral routes depending on their cognitive state, update their attitudes and behaviour, and interact through a Watts-Strogatz social network.

Three demonstration scenarios are included:

| Scenario | What it shows |
|---|---|
| **A** — Route-Dependent Durability | Central-route attitudes persist after campaign ends; peripheral-route attitudes decay rapidly |
| **B** — Feasibility-Constrained Behaviour | Capability and friction interventions unlock behaviour that messaging alone cannot |
| **C** — Amplification and Clustering | Engagement-based visibility amplification increases identity-congruent exposure (filter-bubble effect) |

---

## Requirements

- Python 3.10 or higher
- pip

All other dependencies are installed automatically (see step 2 below).

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/audience-digital-twin.git
cd audience-digital-twin
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the base simulation

```bash
python run.py
```

This runs 60 steps with 200 agents and saves `outputs.csv` and `outputs.png`.

### 4. Run demonstration scenarios

```bash
# Individual scenarios
python run.py --scenario_a
python run.py --scenario_b
python run.py --scenario_c

# All scenarios together
python run.py --scenario_a --scenario_b --scenario_c

# All scenarios plus internal verification tests
python run.py --scenario_a --scenario_b --scenario_c --verify
```

Each scenario saves its own CSV and PNG file:

| Flag | Outputs |
|---|---|
| `--scenario_a` | `scenario_a.csv`, `scenario_a.png` |
| `--scenario_b` | `scenario_b.csv`, `scenario_b.png` |
| `--scenario_c` | `scenario_c.csv`, `scenario_c.png` |
| `--verify` | printed results only |

### 5. Change parameters

All parameters are in `configs/example.yaml`. Edit that file and re-run — no code changes needed. Key parameters to experiment with:

```yaml
population:
  n_agents: 200          # increase for smoother distributions

timing:
  n_steps: 60            # simulation length

dynamics:
  strength_decay: 0.005  # higher = attitudes decay faster (affects Scenario A)
  cap_lr: 0.01           # higher = capability grows faster (affects Scenario B)
  opp_lr: 0.01           # higher = opportunity grows faster

exposure:
  gamma1: 2.0            # amplification strength (0 = off, affects Scenario C)
```

---

## Repository layout

```
audience-digital-twin/
  README.md
  requirements.txt
  pyproject.toml
  configs/
    example.yaml          # all parameters — edit this to change simulation behaviour
  audience_dt/
    __init__.py
    models.py             # Agent, Message, Params, Scenario dataclasses
    sim.py                # simulation engine + population initialisers
    metrics.py            # output aggregation and durability metrics
    verify.py             # internal verification tests
  run.py                  # CLI entrypoint and scenario runner
```

---

## Running in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Upload the repository folder to your Google Drive
2. Open a new Colab notebook and run this setup cell:

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install -r /content/drive/MyDrive/audience-digital-twin/requirements.txt -q

import sys
sys.path.insert(0, '/content/drive/MyDrive/audience-digital-twin')
```

3. Then run scenarios directly in Python:

```python
import yaml, numpy as np, networkx as nx
from audience_dt.models import Params, Scenario
from audience_dt.sim import init_population, simulate
from audience_dt.metrics import outputs_to_frame

# Load config and run — see run.py for full parameter loading helper
```

Or use shell commands:

```python
!python /content/drive/MyDrive/audience-digital-twin/run.py --scenario_a --scenario_b --scenario_c
```

---

## How it works

The model executes six steps per agent per timestep, following Equations 1–15 in the paper:

1. **Exposure** — probability of seeing a message combines peer sharing (network diffusion) with an engagement-to-visibility proxy (platform amplification)
2. **Route selection** — central vs peripheral processing probability depends on need-for-cognition, capability, reflective motivation, and cognitive load
3. **Attitude direction update** — central route uses argument quality; peripheral route uses source credibility, social proof, identity congruence, emotional valence and arousal
4. **Attitude strength update** — central processing produces larger strength gains; high cognitive load reduces consolidation
5. **Intention formation** — linear combination of attitude×strength, motivations, norms, opportunity, and friction
6. **Behaviour** — intention threshold must be met AND capability AND opportunity must exceed their thresholds (COM-B feasibility gate)

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{authors2026audience,
  title     = {A Verifiable Agent-Based Digital Twin Architecture for Audience Response:
               Integrating COM-B and ELM for Durable Behaviour Simulation},
  author    = {[Authors]},
  booktitle = {Proceedings of ICSOFT 2026},
  year      = {2026}
}
```

---

## Licence

MIT
