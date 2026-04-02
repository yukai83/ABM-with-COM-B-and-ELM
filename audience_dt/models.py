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
    nfc: float              # need for cognition
    trust_inst: float       # institutional trust
    trust_peer: float       # peer trust
    identity_salience: float


@dataclass
class AgentState:
    cap: float      # capability (COM-B C)
    opp: float      # opportunity (COM-B O)
    mr: float       # reflective motivation
    ma: float       # automatic motivation
    load: float     # cognitive load
    att: float      # attitude direction [-1, 1]
    strength: float # attitude strength [0, 1]
    norm: float     # descriptive norm
    intent: float
    beh: int
    rep: Dict[int, int]  # per-message repetition count


@dataclass(frozen=True)
class Message:
    msg_id: int
    aq: float       # argument quality
    sc: float       # source credibility
    sp: float       # social proof
    ev: float       # emotional valence
    ea: float       # emotional arousal
    fr: float       # friction
    involve: float = 0.5
    identity_dir: float = 0.0


@dataclass(frozen=True)
class Params:
    alpha: float            # peer vs visibility blend weight
    gamma0: float           # visibility intercept
    gamma1: float           # amplification coefficient (Eq. 3)
    rho_sp: float
    rho_sc: float
    rho_ic: float
    rho_ea: float
    rho_load: float
    w0: float               # route-selection intercept
    w_nfc: float
    w_cap: float
    w_mr: float
    w_load: float
    w_involve: float
    kappa_c: float          # central route scaling
    kappa_p: float          # peripheral route scaling
    a_sc: float
    a_sp: float
    a_ic: float
    a_ev: float
    a_ea: float
    eta_c: float            # central strength gain
    eta_p: float            # peripheral strength gain
    b_sp: float
    lambda_s: float         # load penalty on strength
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
    strength_decay: float = 0.005  # passive decay; controls durability half-life


@dataclass
class Scenario:
    messages_per_step: int = 3
    aq_range: tuple = (0.2, 0.9)
    sc_range: tuple = (0.2, 0.9)
    sp_range: tuple = (0.2, 0.9)
    ev_range: tuple = (-0.6, 0.6)
    ea_range: tuple = (0.0, 1.0)
    fr_range: tuple = (0.0, 1.0)
    involve_range: tuple = (0.2, 0.8)
    identity_dir_range: tuple = (-1.0, 1.0)
    campaign_end_step: Optional[int] = None  # observation phase starts here

    def sample_messages(self, rng: np.random.Generator, step: int, start_id: int) -> List[Message]:
        if self.campaign_end_step is not None and step >= self.campaign_end_step:
            return []

        msgs: List[Message] = []
        for k in range(self.messages_per_step):
            msgs.append(
                Message(
                    msg_id=start_id + k,
                    aq=float(rng.uniform(*self.aq_range)),
                    sc=float(rng.uniform(*self.sc_range)),
                    sp=float(rng.uniform(*self.sp_range)),
                    ev=float(rng.uniform(*self.ev_range)),
                    ea=float(rng.uniform(*self.ea_range)),
                    fr=float(rng.uniform(*self.fr_range)),
                    involve=float(rng.uniform(*self.involve_range)),
                    identity_dir=float(rng.uniform(*self.identity_dir_range)),
                )
            )
        return msgs
