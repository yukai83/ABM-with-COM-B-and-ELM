from __future__ import annotations
from dataclasses import replace
import numpy as np
import networkx as nx
from .models import Params, Scenario
from .sim import init_population, simulate


def verify_route_monotonicity(params: Params, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = 150
    g = nx.watts_strogatz_graph(n=n, k=6, p=0.05, seed=seed)
    traits, states = init_population(n, rng)
    scen = Scenario(messages_per_step=3)

    out_base, _ = simulate(g, traits, states, params, scen, n_steps=20, rng=rng)

    rng2 = np.random.default_rng(seed + 1)
    traits2, states2 = init_population(n, rng2)
    for s in states2.values():
        s.load = 0.9
    out_high, _ = simulate(g, traits2, states2, params, scen, n_steps=20, rng=rng2)

    return {
        "baseline_mean_strength_end": out_base[-1].mean_strength,
        "highload_mean_strength_end": out_high[-1].mean_strength,
        "expectation": "high-load strength <= baseline strength",
        "passed": out_high[-1].mean_strength <= out_base[-1].mean_strength + 1e-6,
    }


def ablation_visibility(params: Params, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = 200
    g = nx.watts_strogatz_graph(n=n, k=6, p=0.05, seed=seed)
    traits, states = init_population(n, rng)
    scen = Scenario(messages_per_step=3)

    out_on, _ = simulate(g, traits, states, params, scen, n_steps=30, rng=rng)

    rng2 = np.random.default_rng(seed + 2)
    traits2, states2 = init_population(n, rng2)
    p_off = replace(params, gamma1=0.0)
    out_off, _ = simulate(g, traits2, states2, p_off, scen, n_steps=30, rng=rng2)

    return {
        "gamma1_on_strength_end": out_on[-1].mean_strength,
        "gamma1_off_strength_end": out_off[-1].mean_strength,
        "gamma1_on_beh_end": out_on[-1].beh_rate,
        "gamma1_off_beh_end": out_off[-1].beh_rate,
        "note": "This is not a prediction test; it checks the ablation runs and produces comparable outputs."
    }
