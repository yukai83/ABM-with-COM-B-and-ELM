# Audience Digital Twin — COM-B + ELM ABM

An agent-based digital twin for simulating audience response to persuasive messaging.
It integrates the **COM-B** behavioural feasibility framework with the **Elaboration Likelihood Model (ELM)** for route-dependent persuasion and attitude durability.

> Companion code for: *"A Verifiable Agent-Based Digital Twin Architecture for Audience Response: Integrating COM-B and ELM for Durable Behaviour Simulation"* — ICSOFT 2026 Position Paper.

---

## What it does

Each agent maintains explicit state variables for capability, opportunity, reflective and automatic motivation, cognitive load, attitude direction, and attitude strength. At every timestep, agents are exposed to messages (via peer diffusion and an engagement-based visibility proxy), process them via central or peripheral routes depending on their cognitive state, update their attitudes and behaviour, and interact through a Watts-Strogatz social network.

Three demonstration scenarios are included:

| Scenario | What it shows |
|---|---|
| **A** — Route-Dependent Durability | Central-route attitudes persist after campaign ends; peripheral-route attitudes decay more quickly |
| **B** — Feasibility-Constrained Behaviour | Capability and friction interventions affect behaviour under the model's feasibility gate |
| **C** — Amplification and Clustering | Engagement-based visibility amplification increases identity-congruent exposure |

---

## Requirements

- Python 3.10 or higher
- pip

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/yukai83/ABM-with-COM-B-and-ELM.git
cd ABM-with-COM-B-and-ELM
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
  cap_lr: 0.01           # capability learning rate
  opp_lr: 0.01           # opportunity learning rate

exposure:
  gamma1: 2.0            # amplification strength (0 = off, affects Scenario C)
```

---

## Repository layout

```text
ABM-with-COM-B-and-ELM/
  README.md
  requirements.txt
  configs/
    example.yaml          # configuration file for the simulation
  audience_dt/
    __init__.py
    models.py             # Agent, Message, Params, Scenario dataclasses
    sim.py                # simulation engine and population initialisers
    metrics.py            # output aggregation and durability metrics
    verify.py             # internal verification tests
  run.py                  # CLI entrypoint and scenario runner
```

---

## Running in Google Colab

1. Upload the repository folder to your Google Drive.
2. Open a new Colab notebook and run this setup cell:

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install -r /content/drive/MyDrive/ABM-with-COM-B-and-ELM/requirements.txt -q

import sys
sys.path.insert(0, '/content/drive/MyDrive/ABM-with-COM-B-and-ELM')
```

3. Then run scenarios directly in Python:

```python
import yaml, numpy as np, networkx as nx
from audience_dt.models import Params, Scenario
from audience_dt.sim import init_population, simulate
from audience_dt.metrics import outputs_to_frame

# Load config and run — see run.py for the full parameter loading helper.
```

Or use shell commands:

```python
!python /content/drive/MyDrive/ABM-with-COM-B-and-ELM/run.py --scenario_a --scenario_b --scenario_c
```

---

## How it works

The model executes six steps per agent per timestep:

1. **Exposure** — probability of seeing a message combines peer sharing (network diffusion) with an engagement-to-visibility proxy (platform amplification)
2. **Route selection** — central vs peripheral processing probability depends on need-for-cognition, capability, reflective motivation, and cognitive load
3. **Attitude direction update** — central route uses argument quality; peripheral route uses source credibility, social proof, identity congruence, and emotional valence
4. **Attitude strength update** — central processing produces larger strength gains; high cognitive load reduces consolidation
5. **Intention formation** — a linear combination of attitude × strength, motivations, norms, opportunity, and friction
6. **Behaviour** — intention threshold must be met and both capability and opportunity must exceed their thresholds

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
