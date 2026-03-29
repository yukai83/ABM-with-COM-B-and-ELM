# Audience Digital Twin — COM-B + ELM ABM (Corrected)

Companion code for:
> *"Toward Verifiable Audience Digital Twins: An Agent-Based Architecture Integrating COM-B and ELM"* — ICSOFT 2026 Position Paper.

This is the **corrected** version of the code, reconciled with the formal model specification in the paper.

---

## Corrections vs. original GitHub release

| File | What changed |
|------|-------------|
| `models.py` | `AgentTraits`: added `pi` (identity preference *P*ᵢ, Table 1, Eq. 14 & Scenario C). `AgentState`: added `fr` (action friction *FR*ᵢ, Table 1, Eq. 26). `Message`: removed `fr` (friction is agent-level); renamed `identity_dir` → `xm` (*X*ₘ, Eq. 13–14). `Params`: added `delta_r`, `delta_a` (δR, δA — pre-message motivation decay, Eq. 4–5). `Scenario`: removed `fr_range`; renamed `identity_dir_range` → `xm_range`. |
| `sim.py` | `identity_congruence()`: *IC*ᵢ,ₘ = 1 − \|*P*ᵢ − *X*ₘ\| / 2 (was using `trust_inst` proxy). `delta_attitude()`: Central route uses κc·AQm·*X*ₘ (Eq. 13); peripheral route uses κp·*X*ₘ·(aSC·*T*inst·SC + aSP·*T*peer·SP + aIC·ID·IC + aEV·EV) (Eq. 14) — added *X*ₘ multiplier and trust weights; removed spurious `cap×mr` term and `aEA` from attitude direction. `p_central()`: uses pre-message decayed *M*r\* (Eq. 12). `simulate()`: pre-message motivation decay (Eq. 4–5); sequential per-message attitude updates; accumulated motivation write-back after loop (Section 4.3). `compute_intent()`: uses agent-level `state.fr` (*FR*ᵢ) and wraps in σ(·) (Eq. 26). Exposure segregation threshold: IC ≥ 0.75 (manuscript Scenario C definition). |
| `run.py` | Scenario A: `xm_range=(1.0, 1.0)` so all messages have *X*ₘ = +1 (Table 2). Scenario B friction: `states[i].fr -= 0.2` (agent-level *FR*ᵢ per Table 1), not message `fr_range`. Scenario C: works via corrected `identity_congruence()`. |
| `configs/example.yaml` | Added `delta_r` and `delta_a` to dynamics block. Removed `fr_range`. |

---

## Quickstart

```bash
pip install -r requirements.txt

# Run base simulation
python run.py

# All demonstration scenarios + verification
python run.py --scenario_a --scenario_b --scenario_c --verify
```

Outputs: `scenario_a.png`, `scenario_b.png`, `scenario_c.png`, plus CSV files.

---

## Repository layout

```
audience_dt/
  __init__.py
  models.py      # AgentTraits, AgentState, Message, Params, Scenario
  sim.py         # simulation engine, population initialisers
  metrics.py     # output aggregation, durability_half_life()
  verify.py      # route monotonicity, IC formula, visibility ablation tests
configs/
  example.yaml   # all parameters
run.py           # CLI entry-point and scenario runners
requirements.txt
```

---

## Corrected scenario results (seed 42)

| Scenario | Metric | Corrected value |
|----------|--------|----------------|
| A — central route | peak strength | 0.840 |
| A — peripheral route | peak strength | 0.298 |
| A — central route | retention at step 49 | 82.1% |
| A — peripheral route | retention at step 49 | 1.1% |
| C — no amplification | mean exposure segregation | 0.260 |
| C — amplification | mean exposure segregation | 0.325 |
| C — G0 (Pᵢ = −1) | end-of-run mean attitude (amplification) | −0.131 |
| C — G1 (Pᵢ = +1) | end-of-run mean attitude (amplification) | +0.359 |

---

## Citation

```bibtex
@inproceedings{zeng2026audience,
  title     = {Toward Verifiable Audience Digital Twins: An Agent-Based
               Architecture Integrating COM-B and ELM},
  author    = {Zeng, Yukai},
  booktitle = {Proceedings of ICSOFT 2026},
  year      = {2026}
}
```

## Licence

MIT
