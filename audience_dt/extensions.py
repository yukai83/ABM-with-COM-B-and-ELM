"""Camera-ready extensions for SIMULTECH 2026 reviewer responses.

This module adds three capabilities requested by reviewers, without altering
the core model or any default behaviour:

  * multi-seed robustness summaries with mean and normal-approximation 95%
    confidence intervals (Reviewer 1, comment 2);
  * one-at-a-time local sensitivity sweeps over influential parameters
    (Reviewer 1, comment 2; verification-first workflow);
  * a minimal diffusion-plus-sentiment baseline and an ablation comparison
    against the full COM-B + ELM model (Reviewer 1, comment 4).

All functions are additive. Existing scenario outputs are unchanged because
the core simulator and its defaults are untouched.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from typing import Callable, Dict, List, Sequence

import numpy as np
import networkx as nx
import pandas as pd

from .models import Params, Scenario
from .sim import init_population, simulate


# ── Summary statistics helper ─────────────────────────────────────────────────

def _summarise(values: Sequence[float]) -> Dict[str, float]:
    """Return mean, sd, n, and a normal-approximation 95% CI half-width.

    A normal approximation is used so that the package keeps its existing
    dependency set (no SciPy required). With a modest number of seeds this is
    adequate for reporting qualitative robustness.
    """
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0,
                "ci95_half": float("nan"), "ci95_lo": float("nan"),
                "ci95_hi": float("nan")}
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    half = 1.96 * sd / np.sqrt(n) if n > 1 else 0.0
    return {"mean": mean, "sd": sd, "n": n,
            "ci95_half": float(half),
            "ci95_lo": float(mean - half),
            "ci95_hi": float(mean + half)}


# ── Multi-seed robustness ─────────────────────────────────────────────────────

def run_multiseed(scenario_fn: Callable[[Params, int], pd.DataFrame],
                  metric_fn: Callable[[pd.DataFrame], Dict[str, float]],
                  params: Params,
                  seeds: Sequence[int]) -> pd.DataFrame:
    """Run ``scenario_fn`` once per seed and summarise scalar metrics.

    Parameters
    ----------
    scenario_fn : callable(params, seed) -> DataFrame
        Any scenario runner (e.g. ``run_scenario_a`` from ``run.py``).
    metric_fn : callable(DataFrame) -> dict[str, float]
        Extracts the scalar diagnostics of interest from one run.
    params : Params
    seeds : sequence of int

    Returns
    -------
    DataFrame indexed by metric name with mean, sd, n, and 95% CI columns.
    """
    per_seed: Dict[str, List[float]] = {}
    for s in seeds:
        df = scenario_fn(params, seed=int(s))
        for k, v in metric_fn(df).items():
            per_seed.setdefault(k, []).append(float(v))

    rows = []
    for metric, vals in per_seed.items():
        summ = _summarise(vals)
        summ["metric"] = metric
        rows.append(summ)
    out = pd.DataFrame(rows).set_index("metric")
    return out[["mean", "sd", "n", "ci95_half", "ci95_lo", "ci95_hi"]]


def scenario_a_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Extract route-durability diagnostics from a Scenario A frame."""
    out: Dict[str, float] = {}
    for route in ("central", "peripheral"):
        sub = df[df["route"] == route]["mean_strength"].to_numpy()
        post = sub[1:] if len(sub) > 1 else sub          # drop the baseline row
        peak = float(post.max()) if len(post) else float("nan")
        end = float(sub[-1]) if len(sub) else float("nan")
        out[f"{route}_peak_strength"] = peak
        out[f"{route}_end_strength"] = end
        out[f"{route}_retention"] = (end / peak) if peak > 0 else float("nan")
    out["retention_gap"] = out.get("central_retention", float("nan")) - \
        out.get("peripheral_retention", float("nan"))
    return out


# ── Local sensitivity (one-at-a-time) ─────────────────────────────────────────

def local_sensitivity(scenario_fn: Callable[[Params, int], pd.DataFrame],
                      metric_fn: Callable[[pd.DataFrame], Dict[str, float]],
                      target_metric: str,
                      params: Params,
                      param_names: Sequence[str],
                      rel_delta: float = 0.10,
                      seed: int = 42) -> pd.DataFrame:
    """One-at-a-time local sensitivity of ``target_metric`` to each parameter.

    Each parameter is perturbed by ``±rel_delta`` (relative). The reported
    elasticity is the symmetric finite-difference derivative scaled by the
    baseline values, giving a unit-free sensitivity score.
    """
    base_metric = metric_fn(scenario_fn(params, seed=seed)).get(target_metric, float("nan"))
    rows = []
    for name in param_names:
        base_val = getattr(params, name)
        if base_val == 0:
            lo_val, hi_val = -abs(rel_delta), abs(rel_delta)
        else:
            lo_val, hi_val = base_val * (1 - rel_delta), base_val * (1 + rel_delta)

        m_lo = metric_fn(scenario_fn(replace(params, **{name: lo_val}), seed=seed)).get(target_metric, float("nan"))
        m_hi = metric_fn(scenario_fn(replace(params, **{name: hi_val}), seed=seed)).get(target_metric, float("nan"))

        d_metric = (m_hi - m_lo)
        d_param = (hi_val - lo_val)
        slope = d_metric / d_param if d_param != 0 else float("nan")
        elasticity = (slope * base_val / base_metric) if base_metric not in (0, float("nan")) else float("nan")
        rows.append({"parameter": name, "base_value": float(base_val),
                     "metric_low": float(m_lo), "metric_high": float(m_hi),
                     "slope": float(slope), "elasticity": float(elasticity)})
    out = pd.DataFrame(rows).set_index("parameter")
    out = out.reindex(out["elasticity"].abs().sort_values(ascending=False).index)
    return out


# ── Diffusion-plus-sentiment ablation baseline ────────────────────────────────

def run_diffusion_sentiment(params: Params, seed: int = 42,
                            n: int = 200, n_steps: int = 50,
                            campaign_end_step: int = 20,
                            update_rate: float = 0.20) -> pd.DataFrame:
    """A deliberately minimal diffusion-plus-single-sentiment baseline.

    Each agent holds a single scalar ``sentiment`` in [-1, 1]. During the
    campaign, exposed agents move their sentiment towards the advocated
    position ``Xm`` at a fixed rate, and also towards the mean sentiment of
    their network neighbours (social diffusion). There is no separate
    attitude-strength state, no ELM route split, and no COM-B feasibility
    gate. This is the ablation reference for the full model: it has no
    mechanism for durable post-campaign retention, so ``mean_strength`` is
    reported as the absolute sentiment level (a stand-in for "commitment").
    """
    rng = np.random.default_rng(seed)
    g = nx.watts_strogatz_graph(n=n, k=6, p=0.05, seed=seed)
    sentiment = rng.uniform(-0.3, 0.3, size=n)
    rows = []
    for t in range(n_steps):
        in_campaign = t < campaign_end_step
        new_sent = sentiment.copy()
        for i in range(n):
            neigh = list(g.neighbors(i))
            social = float(np.mean([sentiment[j] for j in neigh])) if neigh else sentiment[i]
            target = 1.0  # Xm = +1 campaign, mirroring Scenario A
            pull = update_rate * (target - sentiment[i]) if in_campaign else 0.0
            diffusion = 0.10 * (social - sentiment[i])
            new_sent[i] = float(np.clip(sentiment[i] + pull + diffusion, -1.0, 1.0))
        sentiment = new_sent
        rows.append({"t": t, "mean_att": float(sentiment.mean()),
                     "mean_strength": float(np.mean(np.abs(sentiment))),
                     "route": "diffusion_sentiment"})
    return pd.DataFrame(rows)


def run_ablation_baseline(scenario_a_fn: Callable[[Params, int], pd.DataFrame],
                          params: Params, seed: int = 42) -> pd.DataFrame:
    """Compare post-campaign retention: full model routes vs the baseline.

    Returns a small table contrasting durability. The full model should show a
    route-dependent retention gap (central > peripheral); the diffusion-plus-
    sentiment baseline has no route or strength machinery and therefore cannot
    express differential durability.
    """
    df_full = scenario_a_fn(params, seed=seed)
    full = scenario_a_metrics(df_full)

    base_df = run_diffusion_sentiment(params, seed=seed)
    sub = base_df["mean_strength"].to_numpy()
    post = sub[1:] if len(sub) > 1 else sub
    peak = float(post.max()) if len(post) else float("nan")
    end = float(sub[-1]) if len(sub) else float("nan")
    base_ret = (end / peak) if peak > 0 else float("nan")

    rows = [
        {"model": "full (central route)", "peak_strength": full["central_peak_strength"],
         "end_strength": full["central_end_strength"], "retention": full["central_retention"],
         "expresses_route_durability": "yes"},
        {"model": "full (peripheral route)", "peak_strength": full["peripheral_peak_strength"],
         "end_strength": full["peripheral_end_strength"], "retention": full["peripheral_retention"],
         "expresses_route_durability": "yes"},
        {"model": "diffusion+sentiment (ablation)", "peak_strength": peak,
         "end_strength": end, "retention": base_ret,
         "expresses_route_durability": "no (single state, no route/strength)"},
    ]
    return pd.DataFrame(rows).set_index("model")
