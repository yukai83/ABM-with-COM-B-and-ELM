"""Core data structures and helper utilities used by the simulator."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


@dataclass(frozen=True)
class AgentTraits:
    nfc: float                # NFCi
    trust_inst: float         # Tinst,i
    trust_peer: float         # Tpeer,i
    identity_salience: float  # IDi
    pi: float                 # identity preference in [-1, 1]


@dataclass
class AgentState:
    cap: float
    opp: float
    mr: float
    ma: float
    load: float
    att: float
    strength: float
    norm: float
    intent: float
    beh: int
    fr: float        # action friction
    rep: Dict[int, int]


@dataclass(frozen=True)
class Message:
    msg_id: int
    aq: float
    sc: float
    sp: float
    ev: float
    ea: float
    xm: float        # advocated position in [-1, 1]
    involve: float = 0.5


@dataclass(frozen=True)
class Params:
    alpha: float
    gamma0: float
    gamma1: float
    rho_sp: float
    rho_sc: float
    rho_ic: float
    rho_ea: float
    rho_load: float
    w0: float
    w_nfc: float
    w_cap: float
    w_mr: float
    w_load: float
    w_involve: float
    kappa_c: float
    kappa_p: float
    a_sc: float
    a_sp: float
    a_ic: float
    a_ev: float
    a_ea: float
    eta_c: float
    eta_p: float
    b_sp: float
    lambda_s: float
    beta0: float
    beta_as: float
    beta_mr: float
    beta_ma: float
    beta_norm: float
    beta_opp: float
    beta_fric: float
    theta_intent: float
    theta_cap: float
    theta_opp: float
    load_decay: float
    load_from_exposure: float
    norm_mu: float
    mr_lr: float
    ma_lr: float
    cap_lr: float
    opp_lr: float
    strength_decay: float = 0.005
    delta_r: float = 0.005
    delta_a: float = 0.005


@dataclass
class Scenario:
    messages_per_step: int = 3
    aq_range: tuple = (0.2, 0.9)
    sc_range: tuple = (0.2, 0.9)
    sp_range: tuple = (0.2, 0.9)
    ev_range: tuple = (-0.6, 0.6)
    ea_range: tuple = (0.0, 1.0)
    xm_range: tuple = (-1.0, 1.0)
    involve_range: tuple = (0.2, 0.8)
    campaign_end_step: Optional[int] = None

    def sample_messages(self, rng: np.random.Generator, step: int, start_id: int) -> List[Message]:
        if self.campaign_end_step is not None and step >= self.campaign_end_step:
            return []
        lo_xm, hi_xm = self.xm_range
        if lo_xm >= hi_xm:
            hi_xm = lo_xm + 1e-12
        msgs: List[Message] = []
        for k in range(self.messages_per_step):
            msgs.append(Message(
                msg_id=start_id + k,
                aq=float(rng.uniform(*self.aq_range)),
                sc=float(rng.uniform(*self.sc_range)),
                sp=float(rng.uniform(*self.sp_range)),
                ev=float(rng.uniform(*self.ev_range)),
                ea=float(rng.uniform(*self.ea_range)),
                xm=float(rng.uniform(lo_xm, hi_xm)),
                involve=float(rng.uniform(*self.involve_range)),
            ))
        return msgs
