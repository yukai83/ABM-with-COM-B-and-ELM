"""
run.py — Corrected scenario runner.

Key changes from original GitHub version:
  - Scenario A: messages use xm_range=(1.0, 1.0) so all messages have Xm=+1 (Table 2)
  - Scenario B: friction reduction modifies agent-level states[i].fr -= 0.2 (not fr_range)
    [FIXED: FRi is agent-level per Table 1 and Eq.26; original code changed message fr_range]
  - Scenario C: identity_congruence now uses Pi and Xm directly;
                exposure_segregation tracks ICi,m >= 0.75 (manuscript definition)
  - load_params: reads delta_r / delta_a from YAML dynamics block
"""
import argparse
import copy
import yaml
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import replace

from audience_dt.models import Params, Scenario
from audience_dt.sim import init_population, init_population_identity_groups, simulate
from audience_dt.metrics import outputs_to_frame, outputs_to_frame_with_groups, durability_half_life
from audience_dt.verify import verify_route_monotonicity, verify_ic_formula, ablation_visibility


# ── Parameter loading ─────────────────────────────────────────────────────────

def load_params(cfg: dict) -> tuple:
    exp   = cfg["exposure"]
    route = cfg["route"]
    att   = cfg["attitude"]
    st    = cfg["strength"]
    intent = cfg["intention"]
    thr   = cfg["thresholds"]
    dyn   = cfg["dynamics"]
    timing = cfg["timing"]

    p = Params(
        alpha=exp["alpha"], gamma0=exp["gamma0"], gamma1=exp["gamma1"],
        rho_sp=exp["rho"]["sp"], rho_sc=exp["rho"]["sc"],
        rho_ic=exp["rho"]["ic"], rho_ea=exp["rho"]["ea"], rho_load=exp["rho"]["load"],
        w0=route["w0"], w_nfc=route["w_nfc"], w_cap=route["w_cap"],
        w_mr=route["w_mr"], w_load=route["w_load"], w_involve=route["w_involve"],
        kappa_c=att["kappa_c"], kappa_p=att["kappa_p"],
        a_sc=att["a_sc"], a_sp=att["a_sp"], a_ic=att["a_ic"],
        a_ev=att["a_ev"], a_ea=att["a_ea"],
        eta_c=st["eta_c"], eta_p=st["eta_p"], b_sp=st["b_sp"], lambda_s=st["lambda_s"],
        beta0=intent["beta0"], beta_as=intent["beta_as"], beta_mr=intent["beta_mr"],
        beta_ma=intent["beta_ma"], beta_norm=intent["beta_norm"],
        beta_opp=intent["beta_opp"], beta_fric=intent["beta_fric"],
        theta_intent=thr["theta_intent"], theta_cap=thr["theta_cap"], theta_opp=thr["theta_opp"],
        load_decay=dyn["load_decay"], load_from_exposure=dyn["load_from_exposure"],
        norm_mu=dyn["norm_mu"], mr_lr=dyn["mr_lr"], ma_lr=dyn["ma_lr"],
        cap_lr=dyn["cap_lr"], opp_lr=dyn["opp_lr"],
        strength_decay=dyn.get("strength_decay", 0.005),
        delta_r=dyn.get("delta_r", 0.005),   # [ADDED]
        delta_a=dyn.get("delta_a", 0.005),   # [ADDED]
    )
    s = Scenario(messages_per_step=timing["messages_per_step"])
    return p, s, cfg


def build_graph(cfg: dict, seed: int) -> nx.Graph:
    n   = cfg["population"]["n_agents"]
    net = cfg["population"]["network"]
    if net["type"] == "watts_strogatz":
        return nx.watts_strogatz_graph(n=n, k=net["k"], p=net["p_rewire"], seed=seed)
    raise ValueError(f"Unsupported network type: {net['type']}")


def baseline_row_from_states(states: dict, groups: dict | None = None) -> dict:
    """Build a true pre-intervention baseline row for scenario plots."""
    agent_ids = list(states.keys())
    att_arr = np.array([states[i].att for i in agent_ids], dtype=float)
    strength_arr = np.array([states[i].strength for i in agent_ids], dtype=float)
    intent_arr = np.array([states[i].intent for i in agent_ids], dtype=float)
    beh_arr = np.array([states[i].beh for i in agent_ids], dtype=float)

    row = {
        "t": 0,
        "mean_att": float(att_arr.mean()),
        "mean_strength": float(strength_arr.mean()),
        "mean_intent": float(intent_arr.mean()),
        "beh_rate": float(beh_arr.mean()),
        "feasibility_gap": 0.0,
        "att_variance": float(att_arr.var()),
        "exposure_segregation": float("nan"),
    }

    if groups:
        for lbl in sorted(set(groups.values())):
            idx = [i for i in agent_ids if groups[i] == lbl]
            if idx:
                row[f"beh_rate_{lbl}"] = float(np.mean([states[i].beh for i in idx]))
                row[f"mean_att_{lbl}"] = float(np.mean([states[i].att for i in idx]))

    return row


def prepend_baseline(df: pd.DataFrame, baseline_row: dict) -> pd.DataFrame:
    """Shift simulated steps to start at t=1 and prepend a shared t=0 baseline."""
    df_shifted = df.copy()
    df_shifted["t"] = df_shifted["t"] + 1
    return pd.concat([pd.DataFrame([baseline_row]), df_shifted], ignore_index=True, sort=False)


# ── Scenario A ────────────────────────────────────────────────────────────────

def run_scenario_a(params: Params, seed: int = 42) -> pd.DataFrame:
    """
    Two campaigns (central vs peripheral), Xm = +1 in both (Table 2).
    Campaign: 20 steps. Observation phase: 30 steps (no messages).
    Both traces now share a true t=0 baseline before route-favouring conditions begin.
    """
    n = 200
    g = nx.watts_strogatz_graph(n=n, k=6, p=0.05, seed=seed)

    traits_base, states_base = init_population(n, np.random.default_rng(seed))
    baseline_row = baseline_row_from_states(states_base)

    # Central-favouring: high AQ, moderate peripheral cues, low initial load
    traits_c = copy.deepcopy(traits_base)
    states_c = copy.deepcopy(states_base)
    for s in states_c.values():
        s.load = 0.1
    out_c, _ = simulate(g, traits_c, states_c, params,
                        Scenario(messages_per_step=3,
                                 aq_range=(0.7, 0.9),
                                 sc_range=(0.2, 0.4), sp_range=(0.2, 0.4),
                                 ea_range=(0.2, 0.4),
                                 xm_range=(1.0, 1.0),
                                 campaign_end_step=20),
                        n_steps=50, rng=np.random.default_rng(seed))

    # Peripheral-favouring: moderate AQ, strong cues, higher arousal, high initial load
    traits_p = copy.deepcopy(traits_base)
    states_p = copy.deepcopy(states_base)
    for s in states_p.values():
        s.load = 0.6
    out_p, _ = simulate(g, traits_p, states_p, params,
                        Scenario(messages_per_step=3,
                                 aq_range=(0.3, 0.5),
                                 sc_range=(0.7, 0.9), sp_range=(0.7, 0.9),
                                 ea_range=(0.6, 0.8),
                                 xm_range=(1.0, 1.0),
                                 campaign_end_step=20),
                        n_steps=50, rng=np.random.default_rng(seed + 100))

    df_c = prepend_baseline(outputs_to_frame(out_c), baseline_row).assign(route="central")
    df_p = prepend_baseline(outputs_to_frame(out_p), baseline_row).assign(route="peripheral")
    return pd.concat([df_c, df_p], ignore_index=True)


# ── Scenario B ────────────────────────────────────────────────────────────────

def run_scenario_b(params: Params, seed: int = 42) -> pd.DataFrame:
    """
    Four capability quartiles; three conditions:
      baseline, cap_boost_Q1, friction_reduce.
    All three conditions now share a true t=0 baseline before interventions begin.
    """
    n = 200
    n_steps = 60
    g = nx.watts_strogatz_graph(n=n, k=6, p=0.05, seed=seed)

    traits_base, states_base = init_population(n, np.random.default_rng(seed))
    cap_vals = np.array([states_base[i].cap for i in range(n)])
    qthr = np.percentile(cap_vals, [25, 50, 75])
    groups = {}
    for i in range(n):
        c = states_base[i].cap
        if   c <= qthr[0]: groups[i] = "Q1"
        elif c <= qthr[1]: groups[i] = "Q2"
        elif c <= qthr[2]: groups[i] = "Q3"
        else:              groups[i] = "Q4"

    baseline_row = baseline_row_from_states(states_base, groups=groups)
    std_scen = Scenario(messages_per_step=3)
    all_dfs = []

    # Condition 1: Baseline
    states_1 = copy.deepcopy(states_base)
    out_1, _ = simulate(g, copy.deepcopy(traits_base), states_1, params, std_scen,
                        n_steps=n_steps, rng=np.random.default_rng(seed + 1), groups=groups)
    all_dfs.append(prepend_baseline(outputs_to_frame_with_groups(out_1), baseline_row).assign(condition="baseline"))

    # Condition 2: Capability boost for Q1
    states_2 = copy.deepcopy(states_base)
    for i, grp in groups.items():
        if grp == "Q1":
            states_2[i].cap = min(1.0, states_2[i].cap + 0.20)
    out_2, _ = simulate(g, copy.deepcopy(traits_base), states_2, params, std_scen,
                        n_steps=n_steps, rng=np.random.default_rng(seed + 2), groups=groups)
    all_dfs.append(prepend_baseline(outputs_to_frame_with_groups(out_2), baseline_row).assign(condition="cap_boost_Q1"))

    # Condition 3: Friction reduction — modify agent-level FRi
    states_3 = copy.deepcopy(states_base)
    for i in states_3:
        states_3[i].fr = float(np.clip(states_3[i].fr - 0.20, 0.0, 1.0))
    out_3, _ = simulate(g, copy.deepcopy(traits_base), states_3, params, std_scen,
                        n_steps=n_steps, rng=np.random.default_rng(seed + 3), groups=groups)
    all_dfs.append(prepend_baseline(outputs_to_frame_with_groups(out_3), baseline_row).assign(condition="friction_reduce"))

    return pd.concat(all_dfs, ignore_index=True)


# ── Scenario C ────────────────────────────────────────────────────────────────

def run_scenario_c(params: Params, seed: int = 42) -> pd.DataFrame:
    """
    Two identity groups (G0: Pi=-1, G1: Pi=+1), messages span Xm in [-1,1].
    ICi,m = 1 - |Pi - Xm| / 2 (Table 2).
    Exposure segregation = share of processed exposures with ICi,m >= 0.75.
    Both amplification conditions now share a true t=0 baseline.
    """
    n = 200
    n_steps = 60
    g = nx.watts_strogatz_graph(n=n, k=6, p=0.05, seed=seed)
    pol_scen = Scenario(messages_per_step=3, xm_range=(-1.0, 1.0))
    all_dfs = []

    traits_base, states_base, grps = init_population_identity_groups(n, np.random.default_rng(seed))
    baseline_row = baseline_row_from_states(states_base, groups=grps)

    for cond, gamma1_val in [("no_amplification", 0.0), ("amplification", 2.0)]:
        p_cond = replace(params, gamma1=gamma1_val)
        traits_c = copy.deepcopy(traits_base)
        states_c = copy.deepcopy(states_base)
        out, _ = simulate(g, traits_c, states_c, p_cond, pol_scen,
                          n_steps=n_steps, rng=np.random.default_rng(seed),
                          groups=grps, track_segregation=True)
        all_dfs.append(prepend_baseline(outputs_to_frame_with_groups(out), baseline_row).assign(condition=cond))

    return pd.concat(all_dfs, ignore_index=True)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_scenario_a(df: pd.DataFrame, save_path: str = "scenario_a.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Scenario A: Route-Dependent Durability (Xm = +1, campaign ends step 20)",
                 fontweight="bold")
    for route, color in [("central", "steelblue"), ("peripheral", "darkorange")]:
        sub = df[df["route"] == route]
        axes[0].plot(sub["t"], sub["mean_att"],      label=route, color=color)
        axes[1].plot(sub["t"], sub["mean_strength"],  label=route, color=color)
    for ax, title in zip(axes, ["Mean Attitude", "Mean Strength (durability)"]):
        ax.axvline(20, color="grey", ls="--", lw=1, label="campaign end")
        ax.set_xlabel("Step"); ax.set_title(title); ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_scenario_b(df: pd.DataFrame, save_path: str = "scenario_b.png"):
    quartiles  = ["Q1", "Q2", "Q3", "Q4"]
    conditions = ["baseline", "cap_boost_Q1", "friction_reduce"]
    colors     = {"baseline": "steelblue", "cap_boost_Q1": "darkorange", "friction_reduce": "green"}
    fig, axes  = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Scenario B: Feasibility-Constrained Behaviour\n"
                 "Behaviour rate per capability quartile, three conditions",
                 fontsize=12, fontweight="bold")
    for ax, q in zip(axes.flat, quartiles):
        col = f"beh_rate_{q}"
        for cond in conditions:
            sub = df[df["condition"] == cond]
            if col in sub.columns:
                ax.plot(sub["t"], sub[col], label=cond, color=colors[cond])
        ax.set_title(f"Quartile {q}"); ax.set_xlabel("Step"); ax.set_ylabel("Behaviour rate")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_scenario_c(df: pd.DataFrame, save_path: str = "scenario_c.png"):
    colors_cond = {"no_amplification": "steelblue", "amplification": "crimson"}
    colors_grp  = {"G0": "#1f77b4", "G1": "#ff7f0e"}
    fig, axes   = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Scenario C: Amplification and Emergent Clustering\n"
                 "G0 (Pi=−1) vs G1 (Pi=+1)",
                 fontsize=12, fontweight="bold")
    for ax, cond in zip(axes[0], ["amplification", "no_amplification"]):
        sub = df[df["condition"] == cond]
        for grp, color in colors_grp.items():
            col = f"mean_att_{grp}"
            if col in sub.columns:
                ax.plot(sub["t"], sub[col], label=grp, color=color)
        lbl = "amplification on" if cond == "amplification" else "no amplification"
        ax.set_title(f"Group attitudes ({lbl})")
        ax.axhline(0, color="grey", lw=0.8, ls="--"); ax.legend()
    ax = axes[1, 0]
    for cond, color in colors_cond.items():
        sub = df[df["condition"] == cond]
        ax.plot(sub["t"], sub["att_variance"], label=cond, color=color)
    ax.set_title("Attitude variance (polarisation)"); ax.legend()
    ax = axes[1, 1]
    for cond, color in colors_cond.items():
        sub = df[df["condition"] == cond]
        ax.plot(sub["t"], sub["exposure_segregation"], label=cond, color=color)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", label="baseline (0.5)")
    ax.set_title("Exposure segregation\n(ICi,m ≥ 0.75; 0.5 = random)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",    default="configs/example.yaml")
    ap.add_argument("--out_csv",   default="outputs.csv")
    ap.add_argument("--plot",      default="outputs.png")
    ap.add_argument("--verify",    action="store_true")
    ap.add_argument("--scenario_a", action="store_true")
    ap.add_argument("--scenario_b", action="store_true")
    ap.add_argument("--scenario_c", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    params, scenario, cfg = load_params(cfg)
    g    = build_graph(cfg, seed)
    rng  = np.random.default_rng(seed)
    traits, states = init_population(cfg["population"]["n_agents"], rng)
    outputs, _ = simulate(g=g, traits=traits, states=states, params=params,
                          scenario=scenario, n_steps=cfg["timing"]["n_steps"], rng=rng)
    df = outputs_to_frame(outputs)
    df.to_csv(args.out_csv, index=False)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("COM-B + ELM ABM: 200 Agents, 60 Steps", fontsize=13, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0]); ax1.plot(df["t"], df["mean_att"],       color="steelblue");  ax1.set_title("Mean Attitude");  ax1.axhline(0, color="grey", ls="--", lw=0.8)
    ax2 = fig.add_subplot(gs[0, 1]); ax2.plot(df["t"], df["mean_strength"],  color="darkorange"); ax2.set_title("Mean Strength")
    ax3 = fig.add_subplot(gs[1, 0]); ax3.plot(df["t"], df["beh_rate"],       color="green");      ax3.set_title("Behaviour Rate")
    ax4 = fig.add_subplot(gs[1, 1]); ax4.plot(df["t"], df["feasibility_gap"],color="crimson");    ax4.set_title("Feasibility Gap")
    for ax in [ax1, ax2, ax3, ax4]: ax.set_xlabel("Step")
    plt.savefig(args.plot, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved: {args.out_csv}  {args.plot}")
    print(f"Durability half-life: {durability_half_life(df['mean_strength'].to_numpy())}")
    print(f"Final beh_rate={df['beh_rate'].iloc[-1]:.3f}  feasibility_gap={df['feasibility_gap'].iloc[-1]:.3f}  mean_att={df['mean_att'].iloc[-1]:.3f}")

    if args.scenario_a:
        print("\n── Scenario A ──────────────────────────────────────")
        df_a = run_scenario_a(params, seed=seed)
        df_a.to_csv("scenario_a.csv", index=False)
        plot_scenario_a(df_a, "scenario_a.png")
        for route in ["central", "peripheral"]:
            sub = df_a[df_a["route"] == route]["mean_strength"].to_numpy()
            post_baseline = sub[1:] if len(sub) > 1 else sub
            peak   = post_baseline.max()
            peak_t = int(np.argmax(post_baseline)) + 1 if len(post_baseline) else 0
            end    = sub[-1]
            ret    = end / peak if peak > 0 else float("nan")
            hl     = durability_half_life(post_baseline)
            print(f"  {route}: peak={peak:.3f} at step={peak_t}  end={end:.3f}  retention={ret:.1%}  half_life={hl}")

    if args.scenario_b:
        print("\n── Scenario B ──────────────────────────────────────")
        df_b = run_scenario_b(params, seed=seed)
        df_b.to_csv("scenario_b.csv", index=False)
        plot_scenario_b(df_b, "scenario_b.png")
        final_b = df_b[df_b["t"] == df_b["t"].max()]
        for cond in ["baseline", "cap_boost_Q1", "friction_reduce"]:
            row   = final_b[final_b["condition"] == cond].iloc[0]
            rates = {q: f"{row.get(f'beh_rate_{q}', float('nan')):.2f}"
                     for q in ["Q1", "Q2", "Q3", "Q4"]}
            print(f"  {cond:20s}: {rates}")

    if args.scenario_c:
        print("\n── Scenario C ──────────────────────────────────────")
        df_c = run_scenario_c(params, seed=seed)
        df_c.to_csv("scenario_c.csv", index=False)
        plot_scenario_c(df_c, "scenario_c.png")
        for cond in ["no_amplification", "amplification"]:
            sub      = df_c[df_c["condition"] == cond]
            var_end  = sub["att_variance"].iloc[-1]
            mean_seg = sub["exposure_segregation"].mean()
            g0_att   = sub["mean_att_G0"].iloc[-1] if "mean_att_G0" in sub.columns else float("nan")
            g1_att   = sub["mean_att_G1"].iloc[-1] if "mean_att_G1" in sub.columns else float("nan")
            print(f"  {cond:22s}: att_var={var_end:.4f}  mean_seg={mean_seg:.3f}  G0={g0_att:.3f}  G1={g1_att:.3f}")

    if args.verify:
        print("\n── Verification ─────────────────────────────────────")
        print(verify_route_monotonicity(params, seed=seed))
        print(verify_ic_formula(params, seed=seed))
        print(ablation_visibility(params, seed=seed))


if __name__ == "__main__":
    main()
