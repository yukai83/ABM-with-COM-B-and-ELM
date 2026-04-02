from __future__ import annotations
import copy
import numpy as np
import networkx as nx
from .models import Params, Scenario, Message, AgentTraits
from .sim import init_population, simulate, p_central, identity_congruence
from .metrics import outputs_to_frame
from dataclasses import replace


def verify_route_monotonicity(params: Params, seed: int = 42) -> str:
    """High-NFC agents should have higher p_central than low-NFC agents."""
    n = 50
    g = nx.watts_strogatz_graph(n=n, k=4, p=0.05, seed=seed)
    traits, states = init_population(n, np.random.default_rng(seed))
    agent_ids = list(traits.keys())
    msg = Message(msg_id=0, aq=0.5, sc=0.5, sp=0.5, ev=0.0, ea=0.5, xm=0.5, involve=0.5)
    low_nfc_traits = {i: AgentTraits(nfc=0.1, trust_inst=traits[i].trust_inst,
                                     trust_peer=traits[i].trust_peer,
                                     identity_salience=traits[i].identity_salience,
                                     pi=traits[i].pi)
                      for i in agent_ids[:n//2]}
    high_nfc_traits = {i: AgentTraits(nfc=0.9, trust_inst=traits[i].trust_inst,
                                      trust_peer=traits[i].trust_peer,
                                      identity_salience=traits[i].identity_salience,
                                      pi=traits[i].pi)
                       for i in agent_ids[n//2:]}
    low_pc  = float(np.mean([p_central(states[i], low_nfc_traits[i],  msg, params) for i in agent_ids[:n//2]]))
    high_pc = float(np.mean([p_central(states[i], high_nfc_traits[i], msg, params) for i in agent_ids[n//2:]]))
    passed = high_pc > low_pc
    return (f"route_monotonicity: {'PASS' if passed else 'FAIL'} | "
            f"low_NFC_pc={low_pc:.3f}  high_NFC_pc={high_pc:.3f}")


def verify_ic_formula(params: Params, seed: int = 42) -> str:
    """ICi,m = 1 - |Pi - Xm| / 2: fully congruent should give IC=1, fully incongruent IC=0."""
    from .models import AgentTraits
    tr_pos = AgentTraits(nfc=0.5, trust_inst=0.5, trust_peer=0.5, identity_salience=0.7, pi=1.0)
    tr_neg = AgentTraits(nfc=0.5, trust_inst=0.5, trust_peer=0.5, identity_salience=0.7, pi=-1.0)
    msg_pos = Message(msg_id=0, aq=0.5, sc=0.5, sp=0.5, ev=0.0, ea=0.5, xm=1.0)
    msg_neg = Message(msg_id=1, aq=0.5, sc=0.5, sp=0.5, ev=0.0, ea=0.5, xm=-1.0)
    ic_pp = identity_congruence(tr_pos, msg_pos)  # should be 1.0
    ic_pn = identity_congruence(tr_pos, msg_neg)  # should be 0.0
    ic_np = identity_congruence(tr_neg, msg_pos)  # should be 0.0
    ic_nn = identity_congruence(tr_neg, msg_neg)  # should be 1.0
    ok = (abs(ic_pp - 1.0) < 1e-9 and abs(ic_pn) < 1e-9 and
          abs(ic_np) < 1e-9 and abs(ic_nn - 1.0) < 1e-9)
    return (f"ic_formula: {'PASS' if ok else 'FAIL'} | "
            f"ic(+,+)={ic_pp:.3f}  ic(+,-)={ic_pn:.3f}  ic(-,+)={ic_np:.3f}  ic(-,-)={ic_nn:.3f}")


def ablation_visibility(params: Params, seed: int = 42) -> str:
    """gamma1=2 should produce higher mean attitude than gamma1=0 (engagement amplification)."""
    n = 100
    g = nx.watts_strogatz_graph(n=n, k=4, p=0.05, seed=seed)
    scenario = Scenario(messages_per_step=2)
    p0 = replace(params, gamma1=0.0)
    p2 = replace(params, gamma1=2.0)
    traits0, states0 = init_population(n, np.random.default_rng(seed))
    traits2, states2 = init_population(n, np.random.default_rng(seed))
    out0, _ = simulate(g, traits0, states0, p0, scenario, n_steps=20, rng=np.random.default_rng(seed))
    out2, _ = simulate(g, traits2, states2, p2, scenario, n_steps=20, rng=np.random.default_rng(seed))
    df0 = outputs_to_frame(out0)
    df2 = outputs_to_frame(out2)
    return (f"ablation_visibility: gamma1=0 mean_att={df0['mean_att'].mean():.3f} | "
            f"gamma1=2 mean_att={df2['mean_att'].mean():.3f}")
