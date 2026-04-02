from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from .models import AgentTraits, AgentState, Message, Params, Scenario, sigmoid, clip


# ---------------------------------------------------------------------------
# Helper functions (Equations 1-15 in paper)
# ---------------------------------------------------------------------------

def identity_congruence(traits: AgentTraits, msg: Message) -> float:
    """Maps trust_inst to an agent identity direction, then computes
    congruence with the message. Scaled by identity_salience so low-salience
    agents converge toward 0.5 regardless of content."""
    agent_dir = 2.0 * traits.trust_inst - 1.0
    raw = 1.0 - abs(agent_dir - msg.identity_dir) / 2.0
    ic01 = traits.identity_salience * raw + (1.0 - traits.identity_salience) * 0.5
    return float(2.0 * ic01 - 1.0)


def peer_component(g: nx.Graph, shares_prev: Dict[int, List[int]], i: int, msg_id: int) -> float:
    neigh = list(g.neighbors(i))
    if not neigh:
        return 0.0
    sharers = set(shares_prev.get(msg_id, []))
    count = sum((n in sharers) for n in neigh)
    return count / len(neigh)


def engage_propensity(state: AgentState, traits: AgentTraits, msg: Message, p: Params) -> float:
    ic = identity_congruence(traits, msg)
    return (
        p.rho_sp * msg.sp
        + p.rho_sc * msg.sc
        + p.rho_ic * ic
        + p.rho_ea * msg.ea
        - p.rho_load * state.load
    )


def visibility(state: AgentState, traits: AgentTraits, msg: Message, p: Params) -> float:
    eng = engage_propensity(state, traits, msg, p)
    return sigmoid(p.gamma0 + p.gamma1 * eng)


def exposure_prob(g, shares_prev, i, state, traits, msg, p):
    peer = peer_component(g, shares_prev, i, msg.msg_id)
    vis = visibility(state, traits, msg, p)
    return clip(p.alpha * peer + (1.0 - p.alpha) * vis, 0.0, 1.0)


def p_central(state: AgentState, traits: AgentTraits, msg: Message, p: Params) -> float:
    z = (
        p.w0
        + p.w_nfc * traits.nfc
        + p.w_cap * state.cap
        + p.w_mr  * state.mr
        - p.w_load * state.load
        + p.w_involve * msg.involve
    )
    return float(sigmoid(z))


def delta_attitude(state, traits, msg, pcent, p):
    ic = identity_congruence(traits, msg)

    f_c = p.kappa_c * msg.aq * state.cap * state.mr
    cue = (
        p.a_sc * msg.sc
        + p.a_sp * msg.sp
        + p.a_ic * ic
        + p.a_ev * msg.ev
        + p.a_ea * msg.ea
    )
    f_p = p.kappa_p * cue
    raw = pcent * f_c + (1.0 - pcent) * f_p

    # ceiling/floor resistance: harder to push agents already near extremes
    direction = 1.0 if raw >= 0.0 else -1.0
    resistance = 1.0 - max(0.0, state.att * direction)
    return raw * resistance


def delta_strength(state, traits, msg, pcent, p):
    rep = state.rep.get(msg.msg_id, 0)
    g_c = p.eta_c * msg.aq
    g_p = p.eta_p * (np.log1p(rep) + p.b_sp * msg.sp)
    return float(pcent * g_c + (1.0 - pcent) * g_p - p.lambda_s * state.load)


def compute_intent(state, msg_fric, p):
    return (
        p.beta0
        + p.beta_as   * state.att * state.strength
        + p.beta_mr   * state.mr
        + p.beta_ma   * state.ma
        + p.beta_norm * state.norm
        + p.beta_opp  * state.opp
        - p.beta_fric * msg_fric
    )


def behaviour_from_intent(state, intent, p):
    return int(
        (intent > p.theta_intent)
        and (state.cap > p.theta_cap)
        and (state.opp > p.theta_opp)
    )


# ---------------------------------------------------------------------------
# Step-level output container
# ---------------------------------------------------------------------------

@dataclass
class StepOutputs:
    mean_att: float
    mean_strength: float
    mean_intent: float
    beh_rate: float
    feasibility_gap: float
    shares: Dict[int, List[int]]
    att_variance: float = 0.0
    group_beh_rates: Dict[str, float] = field(default_factory=dict)
    group_mean_atts: Dict[str, float] = field(default_factory=dict)
    exposure_segregation: float = float("nan")


# ---------------------------------------------------------------------------
# Core simulation loop
# ---------------------------------------------------------------------------

def simulate(
    g,
    traits,
    states,
    params,
    scenario,
    n_steps,
    rng,
    groups: Optional[Dict[int, str]] = None,
    track_segregation: bool = False,
):
    outputs = []
    shares_prev = {}
    next_msg_id = 0

    group_labels: List[str] = sorted(set(groups.values())) if groups else []

    for t in range(n_steps):
        msgs = scenario.sample_messages(rng=rng, step=t, start_id=next_msg_id)
        next_msg_id += len(msgs)

        shares_now = {m.msg_id: [] for m in msgs}
        beh_prev = np.array([states[i].beh for i in states.keys()], dtype=float)
        global_beh_rate = float(beh_prev.mean()) if len(beh_prev) else 0.0

        feasibility_gap_count = 0
        intent_high_count = 0
        seg_aligned = 0
        seg_total = 0

        for i in states.keys():
            st = states[i]
            tr = traits[i]

            # per-step dynamics: decay load, update norm, decay strength, grow cap/opp
            st.load     = clip(st.load    - params.load_decay, 0.0, 1.0)
            st.norm     = clip((1.0 - params.norm_mu) * st.norm + params.norm_mu * global_beh_rate, 0.0, 1.0)
            st.strength = clip(st.strength - params.strength_decay, 0.0, 1.0)
            st.cap      = clip(st.cap     + params.cap_lr * st.mr,   0.0, 1.0)
            st.opp      = clip(st.opp     + params.opp_lr * st.norm, 0.0, 1.0)

            # exposure
            exposed_msgs = []
            for m in msgs:
                p_exp = exposure_prob(g, shares_prev, i, st, tr, m, params)
                if rng.uniform() < p_exp:
                    exposed_msgs.append(m)
                    st.load = clip(st.load + params.load_from_exposure, 0.0, 1.0)

                    if track_segregation:
                        seg_total += 1
                        if identity_congruence(tr, m) > 0:
                            seg_aligned += 1

            # attitude and strength updates
            dA_total = 0.0
            dS_total = 0.0
            fr_bar = 0.0

            for m in exposed_msgs:
                pc  = p_central(st, tr, m, params)
                dA  = delta_attitude(st, tr, m, pc, params)
                dS  = delta_strength(st, tr, m, pc, params)
                dA_total += dA
                dS_total += dS
                fr_bar   += m.fr
                st.rep[m.msg_id] = st.rep.get(m.msg_id, 0) + 1
                st.mr = clip(st.mr + params.mr_lr * (pc * m.aq), 0.0, 1.0)
                st.ma = clip(st.ma + params.ma_lr * ((1.0 - pc) * (m.ea + m.sp) / 2.0), 0.0, 1.0)

                eng = engage_propensity(st, tr, m, params)
                if eng > 0.8:
                    shares_now[m.msg_id].append(i)

            if exposed_msgs:
                fr_bar /= max(1, len(exposed_msgs))

            st.att      = clip(st.att + dA_total, -1.0, 1.0)
            st.strength = clip(st.strength + dS_total, 0.0, 1.0)
            st.intent   = compute_intent(st, fr_bar, params)
            st.beh      = behaviour_from_intent(st, st.intent, params)

            if st.intent > params.theta_intent:
                intent_high_count += 1
                if (st.cap <= params.theta_cap) or (st.opp <= params.theta_opp):
                    feasibility_gap_count += 1

        # aggregate
        agent_ids   = list(states.keys())
        att_arr     = np.array([states[i].att      for i in agent_ids], dtype=float)
        strength_arr= np.array([states[i].strength for i in agent_ids], dtype=float)
        intent_arr  = np.array([states[i].intent   for i in agent_ids], dtype=float)
        beh_arr     = np.array([states[i].beh      for i in agent_ids], dtype=float)

        group_beh_rates: Dict[str, float] = {}
        group_mean_atts: Dict[str, float] = {}
        if groups:
            for lbl in group_labels:
                idx = [i for i in agent_ids if groups[i] == lbl]
                if idx:
                    group_beh_rates[lbl] = float(np.mean([states[i].beh for i in idx]))
                    group_mean_atts[lbl] = float(np.mean([states[i].att for i in idx]))

        exp_seg = float(seg_aligned / seg_total) if seg_total > 0 else float("nan")

        outputs.append(StepOutputs(
            mean_att=float(att_arr.mean()),
            mean_strength=float(strength_arr.mean()),
            mean_intent=float(intent_arr.mean()),
            beh_rate=float(beh_arr.mean()),
            feasibility_gap=float(feasibility_gap_count / max(1, intent_high_count)),
            shares=shares_now,
            att_variance=float(att_arr.var()),
            group_beh_rates=group_beh_rates,
            group_mean_atts=group_mean_atts,
            exposure_segregation=exp_seg,
        ))

        shares_prev = shares_now

    return outputs, states


# ---------------------------------------------------------------------------
# Population initialisers
# ---------------------------------------------------------------------------

def init_population(n_agents, rng):
    from .models import AgentTraits, AgentState
    traits = {}
    states = {}
    for i in range(n_agents):
        traits[i] = AgentTraits(
            nfc=float(rng.uniform(0.2, 0.8)),
            trust_inst=float(rng.uniform(0.2, 0.8)),
            trust_peer=float(rng.uniform(0.2, 0.8)),
            identity_salience=float(rng.uniform(0.2, 0.8)),
        )
        states[i] = AgentState(
            cap=float(rng.uniform(0.3, 0.8)),
            opp=float(rng.uniform(0.3, 0.8)),
            mr=float(rng.uniform(0.3, 0.8)),
            ma=float(rng.uniform(0.3, 0.8)),
            load=float(rng.uniform(0.0, 0.4)),
            att=float(rng.uniform(-0.3, 0.3)),
            strength=float(rng.uniform(0.1, 0.5)),
            norm=float(rng.uniform(0.2, 0.8)),
            intent=0.0,
            beh=0,
            rep={},
        )
    return traits, states


def init_population_identity_groups(n_agents: int, rng: np.random.Generator):
    """Two-group initialiser for Scenario C.

    G0 agents have low institutional trust (and thus a negative identity direction),
    G1 agents have high trust. Both groups have elevated identity_salience so the
    split actually bites. Returns traits, states, and a group membership dict.
    """
    from .models import AgentTraits, AgentState
    traits  = {}
    states  = {}
    groups  = {}
    half = n_agents // 2

    for i in range(n_agents):
        grp = "G0" if i < half else "G1"
        groups[i] = grp
        trust_inst = float(
            rng.uniform(0.10, 0.35) if grp == "G0" else rng.uniform(0.65, 0.90)
        )
        traits[i] = AgentTraits(
            nfc=float(rng.uniform(0.2, 0.8)),
            trust_inst=trust_inst,
            trust_peer=float(rng.uniform(0.2, 0.8)),
            identity_salience=float(rng.uniform(0.55, 0.85)),
        )
        states[i] = AgentState(
            cap=float(rng.uniform(0.3, 0.8)),
            opp=float(rng.uniform(0.3, 0.8)),
            mr=float(rng.uniform(0.3, 0.8)),
            ma=float(rng.uniform(0.3, 0.8)),
            load=float(rng.uniform(0.0, 0.4)),
            att=float(rng.uniform(-0.3, 0.3)),
            strength=float(rng.uniform(0.1, 0.5)),
            norm=float(rng.uniform(0.2, 0.8)),
            intent=0.0,
            beh=0,
            rep={},
        )

    return traits, states, groups
