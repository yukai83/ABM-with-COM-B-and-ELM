"""
sim.py — Corrected simulation engine.

Key changes from original GitHub version:
  1. identity_congruence(): ICi,m = 1 - |Pi - Xm| / 2  (Eq. per Table 2/Scenario C spec)
     Uses traits.pi and msg.xm. Returns [0,1].
  2. delta_attitude():
     - Central: ΔA^c = κc * AQm * Xm         (Eq.13 — was missing Xm, had spurious cap*mr)
     - Peripheral: ΔA^p = κp * Xm * (aSC*Tinst*SCm + aSP*Tpeer*SPm + aIC*IDi*ICi,m + aEV*EVm)
       (Eq.14 — added Xm, added trust multipliers, removed aEA which is not in Eq.14)
  3. p_central(): accepts mr_star param so route uses pre-message decayed M^r* (Eq.12)
  4. simulate(): pre-message motivation decay (Eq.4-5), sequential per-message attitude/strength
     updates, accumulated motivation gains written back after loop (Section 4.3).
  5. compute_intent(): uses agent-level FRi (state.fr, not message friction); applies sigmoid (Eq.26)
  6. Exposure segregation: counts ICi,m >= 0.75 (manuscript Scenario C definition)
  7. init_population(): initialises pi and fr for each agent
  8. init_population_identity_groups(): sets pi=-1 (G0) / pi=+1 (G1)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import networkx as nx
from .models import AgentTraits, AgentState, Message, Params, Scenario, sigmoid, clip


# ── Per-message helper functions ──────────────────────────────────────────────

def identity_congruence(traits: AgentTraits, msg: Message) -> float:
    """
    ICi,m = 1 - |Pi - Xm| / 2  (Table 2, Scenario C definition)
    Returns value in [0, 1]: 1 = fully congruent, 0 = fully incongruent.
    [FIXED: was deriving Pi from trust_inst and returning [-1,1]]
    """
    return float(1.0 - abs(traits.pi - msg.xm) / 2.0)


def peer_component(g: nx.Graph, shares_prev: Dict[int, List[int]], i: int, msg_id: int) -> float:
    neigh = list(g.neighbors(i))
    if not neigh:
        return 0.0
    sharers = set(shares_prev.get(msg_id, []))
    return sum(n in sharers for n in neigh) / len(neigh)


def engage_propensity(state: AgentState, traits: AgentTraits, msg: Message, p: Params) -> float:
    """Eq.9: Engagei,m = ρSP*SPm + ρSC*SCm + ρIC*ICi,m + ρEA*EAm - ρL*Li"""
    ic = identity_congruence(traits, msg)   # now in [0,1]
    return (
        p.rho_sp * msg.sp
        + p.rho_sc * msg.sc
        + p.rho_ic * ic
        + p.rho_ea * msg.ea
        - p.rho_load * state.load
    )


def visibility(state: AgentState, traits: AgentTraits, msg: Message, p: Params) -> float:
    """Eq.10: Visi,m = σ(γ0 + γ1*Engagei,m)"""
    return sigmoid(p.gamma0 + p.gamma1 * engage_propensity(state, traits, msg, p))


def exposure_prob(g, shares_prev, i, state, traits, msg, p) -> float:
    """Eq.11: Ei,m = clip(α*Peer + (1-α)*Vis, 0, 1)"""
    peer = peer_component(g, shares_prev, i, msg.msg_id)
    vis = visibility(state, traits, msg, p)
    return clip(p.alpha * peer + (1.0 - p.alpha) * vis, 0.0, 1.0)


def p_central(state: AgentState, traits: AgentTraits, msg: Message, p: Params,
              mr_star: Optional[float] = None) -> float:
    """
    Eq.12: p^c_i,m = σ(w0 + wNFC*NFCi + wC*Ci + wR*M^r*_i - wL*Li + wI*Ii,m)
    Uses mr_star (pre-message decayed M^r*) when provided.
    [FIXED: now accepts mr_star to enforce use of pre-message decayed value]
    """
    mr = mr_star if mr_star is not None else state.mr
    z = (
        p.w0
        + p.w_nfc * traits.nfc
        + p.w_cap * state.cap
        + p.w_mr * mr
        - p.w_load * state.load
        + p.w_involve * msg.involve
    )
    return float(sigmoid(z))


def delta_attitude(state: AgentState, traits: AgentTraits, msg: Message,
                   pcent: float, p: Params) -> float:
    """
    Eq.13-17: attitude direction update.
    Central:    ΔA^c = κc * AQm * Xm
    Peripheral: ΔA^p = κp * Xm * (aSC*Tinst*SCm + aSP*Tpeer*SPm + aIC*IDi*ICi,m + aEV*EVm)
    Combined and resistance-gated.
    [FIXED: added Xm to both routes; added trust multipliers to peripheral;
            removed aEA from peripheral attitude direction (EAm not in Eq.14)]
    """
    ic = identity_congruence(traits, msg)

    # Eq.13 — central: κc * AQm * Xm
    f_c = p.kappa_c * msg.aq * msg.xm

    # Eq.14 — peripheral: κp * Xm * (aSC*Tinst*SCm + aSP*Tpeer*SPm + aIC*IDi*ICi,m + aEV*EVm)
    cue = (
        p.a_sc * traits.trust_inst * msg.sc
        + p.a_sp * traits.trust_peer * msg.sp
        + p.a_ic * traits.identity_salience * ic
        + p.a_ev * msg.ev
    )
    f_p = p.kappa_p * msg.xm * cue

    # Eq.15 — weighted mix
    raw = pcent * f_c + (1.0 - pcent) * f_p

    # Eq.16-17 — resistance prevents boundary saturation
    direction = float(np.sign(raw)) if raw != 0 else 0.0
    resistance = 1.0 - max(0.0, state.att * direction)
    return raw * resistance


def delta_strength(state: AgentState, traits: AgentTraits, msg: Message,
                   pcent: float, p: Params) -> float:
    """
    Eq.18-20: attitude strength update (unchanged from manuscript spec).
    g^c = ηc * AQm
    g^p = ηp * (ln(1 + repi,m) + bSP*SPm)
    ΔS  = p^c*g^c + (1-p^c)*g^p - λs*Li
    """
    rep = state.rep.get(msg.msg_id, 0)
    g_c = p.eta_c * msg.aq
    g_p = p.eta_p * (np.log1p(rep) + p.b_sp * msg.sp)
    return float(pcent * g_c + (1.0 - pcent) * g_p - p.lambda_s * state.load)


def compute_intent(state: AgentState, p: Params) -> float:
    """
    Eq.26: BIi = σ(β0 + βAS*Ai*Si + βR*M^r_i + βA*M^a_i + βN*Ni + βO*Oi - βF*FRi)
    [FIXED: now uses agent-level state.fr (FRi) instead of message-level friction;
            applies sigmoid as per Eq.26]
    """
    raw = (
        p.beta0
        + p.beta_as * state.att * state.strength
        + p.beta_mr * state.mr
        + p.beta_ma * state.ma
        + p.beta_norm * state.norm
        + p.beta_opp * state.opp
        - p.beta_fric * state.fr
    )
    return float(sigmoid(raw))


def behaviour_from_intent(state: AgentState, intent: float, p: Params) -> int:
    """Eq.27: Bi = 1[BIi >= θI ∧ Ci >= θC ∧ Oi >= θO]"""
    return int(
        (intent >= p.theta_intent)
        and (state.cap >= p.theta_cap)
        and (state.opp >= p.theta_opp)
    )


# ── Step output dataclass ─────────────────────────────────────────────────────

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


# ── Core simulation loop ──────────────────────────────────────────────────────

def simulate(g, traits, states, params, scenario, n_steps, rng,
             groups=None, track_segregation=False):
    """
    Main simulation loop. Implements Section 4.3 implementation order:
      (i)   slow-state decay + lagged population norm
      (ii)  pre-message motivation decay → mr_star, ma_star  [FIXED]
      (iii) exposure probabilities
      (iv)  per-message: route, attitude update (sequential), strength update,
            motivation gain accumulation  [FIXED: sequential attitude; accumulated mr/ma]
      (v)   motivation write-back after loop  [FIXED]
      (vi)  intention (sigmoid) + behaviour  [FIXED]
      (vii) sharing for t+1 diffusion
    """
    outputs = []
    shares_prev: Dict[int, List[int]] = {}
    next_msg_id = 0
    group_labels: List[str] = sorted(set(groups.values())) if groups else []

    for t in range(n_steps):
        msgs = scenario.sample_messages(rng=rng, step=t, start_id=next_msg_id)
        next_msg_id += len(msgs)
        shares_now: Dict[int, List[int]] = {m.msg_id: [] for m in msgs}

        # Lagged population behaviour rate (used in norm update)
        beh_prev = np.array([states[i].beh for i in states], dtype=float)
        global_beh_rate = float(beh_prev.mean()) if len(beh_prev) else 0.0

        feasibility_gap_count = 0
        intent_high_count = 0
        seg_aligned = 0
        seg_total = 0

        for i in states:
            st = states[i]
            tr = traits[i]

            # ── (i) Slow-state updates (Eq.1-3, 6-7) ────────────────────────
            st.load     = clip(st.load - params.load_decay, 0.0, 1.0)
            st.norm     = clip((1.0 - params.norm_mu) * st.norm + params.norm_mu * global_beh_rate, 0.0, 1.0)
            st.strength = clip(st.strength - params.strength_decay, 0.0, 1.0)
            st.cap      = clip(st.cap + params.cap_lr * st.mr, 0.0, 1.0)
            st.opp      = clip(st.opp + params.opp_lr * st.norm, 0.0, 1.0)

            # ── (ii) Pre-message motivation decay → mr_star, ma_star (Eq.4-5) [FIXED]
            mr_star = clip(st.mr - params.delta_r, 0.0, 1.0)
            ma_star = clip(st.ma - params.delta_a, 0.0, 1.0)

            # ── (iii) Exposure ────────────────────────────────────────────────
            exposed_msgs = []
            for m in msgs:
                p_exp = exposure_prob(g, shares_prev, i, st, tr, m, params)
                if rng.uniform() < p_exp:
                    exposed_msgs.append(m)
                    st.load = clip(st.load + params.load_from_exposure, 0.0, 1.0)
                    if track_segregation:
                        ic_val = identity_congruence(tr, m)
                        seg_total += 1
                        if ic_val >= 0.75:   # manuscript Scenario C definition [FIXED]
                            seg_aligned += 1

            # ── (iv) Per-message: route, attitude (sequential), strength, motivation ──
            delta_mr_acc = 0.0
            delta_ma_acc = 0.0

            for m in exposed_msgs:
                # Route probability uses mr_star (Eq.12) [FIXED]
                pc = p_central(st, tr, m, params, mr_star=mr_star)

                # Attitude update applied immediately (sequential, Eq.15-17) [FIXED]
                dA = delta_attitude(st, tr, m, pc, params)
                st.att = clip(st.att + dA, -1.0, 1.0)

                # Strength update applied immediately (Eq.18-21)
                dS = delta_strength(st, tr, m, pc, params)
                st.strength = clip(st.strength + dS, 0.0, 1.0)

                # Accumulate motivation gains (Eq.22-23) [FIXED: accumulate, not in-loop write]
                delta_mr_acc += params.mr_lr * pc * m.aq
                delta_ma_acc += params.ma_lr * (1.0 - pc) * (m.ea + m.sp) / 2.0

                # Repetition counter (incremented after processing, Section 4.3 step iv)
                st.rep[m.msg_id] = st.rep.get(m.msg_id, 0) + 1

                # Sharing: if engagement > τshare
                eng = engage_propensity(st, tr, m, params)
                if eng > 0.8:
                    shares_now[m.msg_id].append(i)

            # ── (v) Write back motivations after loop (Eq.24-25) [FIXED] ─────
            st.mr = clip(mr_star + delta_mr_acc, 0.0, 1.0)
            st.ma = clip(ma_star + delta_ma_acc, 0.0, 1.0)

            # ── (vi) Intention + behaviour (Eq.26-27) [FIXED: sigmoid, agent FRi] ──
            st.intent = compute_intent(st, params)
            st.beh    = behaviour_from_intent(st, st.intent, params)

            if st.intent >= params.theta_intent:
                intent_high_count += 1
                if (st.cap < params.theta_cap) or (st.opp < params.theta_opp):
                    feasibility_gap_count += 1

        # ── Population-level aggregates ────────────────────────────────────
        agent_ids = list(states.keys())
        att_arr      = np.array([states[i].att      for i in agent_ids], dtype=float)
        strength_arr = np.array([states[i].strength  for i in agent_ids], dtype=float)
        intent_arr   = np.array([states[i].intent    for i in agent_ids], dtype=float)
        beh_arr      = np.array([states[i].beh       for i in agent_ids], dtype=float)

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


# ── Population initialisers ───────────────────────────────────────────────────

def init_population(n_agents: int, rng: np.random.Generator):
    """
    Standard heterogeneous population (Table 1 initialisation ranges).
    [FIXED: now initialises pi and fr for each agent]
    """
    traits = {}
    states = {}
    for i in range(n_agents):
        traits[i] = AgentTraits(
            nfc=float(rng.uniform(0.2, 0.8)),
            trust_inst=float(rng.uniform(0.2, 0.8)),
            trust_peer=float(rng.uniform(0.2, 0.8)),
            identity_salience=float(rng.uniform(0.2, 0.8)),
            pi=float(rng.uniform(-1.0, 1.0)),    # Pi ~ U(-1,1) per Table 1
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
            fr=float(rng.uniform(0.2, 0.8)),   # FRi ~ U(0.2,0.8) per Table 1 [ADDED]
            rep={},
        )
    return traits, states


def init_population_identity_groups(n_agents: int, rng: np.random.Generator):
    """
    Scenario C population: two identity groups.
    G0: Pi = -1, lower institutional trust
    G1: Pi = +1, higher institutional trust
    [FIXED: sets pi=-1 (G0) and pi=+1 (G1) per Scenario C spec in Table 2;
            now initialises fr for each agent]
    """
    traits = {}
    states = {}
    groups = {}
    half = n_agents // 2

    for i in range(n_agents):
        grp = "G0" if i < half else "G1"
        groups[i] = grp

        if grp == "G0":
            trust_inst = float(rng.uniform(0.10, 0.35))
            pi_val = -1.0    # G0: Pi = -1 (Table 2 Scenario C)
        else:
            trust_inst = float(rng.uniform(0.65, 0.90))
            pi_val = 1.0     # G1: Pi = +1 (Table 2 Scenario C)

        traits[i] = AgentTraits(
            nfc=float(rng.uniform(0.2, 0.8)),
            trust_inst=trust_inst,
            trust_peer=float(rng.uniform(0.2, 0.8)),
            identity_salience=float(rng.uniform(0.55, 0.85)),
            pi=pi_val,       # [FIXED: explicit Pi per manuscript spec]
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
            fr=float(rng.uniform(0.2, 0.8)),
            rep={},
        )

    return traits, states, groups
