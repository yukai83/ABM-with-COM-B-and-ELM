import argparse
import copy
import yaml
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import replace

from audience_dt.models import Params, Scenario
from audience_dt.sim import init_population, init_population_identity_groups, simulate
from audience_dt.metrics import outputs_to_frame, outputs_to_frame_with_groups, durability_half_life
from audience_dt.verify import verify_route_monotonicity, ablation_visibility


# ── Parameter loading ─────────────────────────────────────────────────────────

def load_params(cfg: dict) -> tuple[Params, Scenario, dict]:
    exp = cfg["exposure"]
    route = cfg["route"]
    att = cfg["attitude"]
    strength = cfg["strength"]
    intent = cfg["intention"]
    thr = cfg["thresholds"]
    dyn = cfg["dynamics"]
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
        eta_c=strength["eta_c"], eta_p=strength["eta_p"],
        b_sp=strength["b_sp"], lambda_s=strength["lambda_s"],
        beta0=intent["beta0"], beta_as=intent["beta_as"], beta_mr=intent["beta_mr"],
        beta_ma=intent["beta_ma"], beta_norm=intent["beta_norm"],
        beta_opp=intent["beta_opp"], beta_fric=intent["beta_fric"],
        theta_intent=thr["theta_intent"], theta_cap=thr["theta_cap"], theta_opp=thr["theta_opp"],
        load_decay=dyn["load_decay"], load_from_exposure=dyn["load_from_exposure"],
        norm_mu=dyn["norm_mu"], mr_lr=dyn["mr_lr"], ma_lr=dyn["ma_lr"],
        cap_lr=dyn["cap_lr"], opp_lr=dyn["opp_lr"],
        strength_decay=dyn.get("strength_decay", 0.005),
    )
    s = Scenario(messages_per_step=timing["messages_per_step"])
    return p, s, cfg


def build_graph(cfg: dict, seed: int) -> nx.Graph:
    n = cfg["population"]["n_agents"]
    net = cfg["population"]["network"]
    if net["type"] == "watts_strogatz":
        return nx.watts_strogatz_graph(n=n, k=net["k"], p=net["p_rewire"], seed=seed)
    raise ValueError(f"Unsupported network type: {net['type']}")


# ── Scenario A: Route-Dependent Durability ────────────────────────────────────

def run_scenario_a(params: Params, seed: int = 42) -> pd.DataFrame:
    """
    Two campaigns (central-favoring vs peripheral-favoring), each 20 steps,
    followed by a 30-step observation/decay phase with no new messages.
    Demonstrates that central-route attitudes retain strength longer.
    """
    n = 200
    g = nx.watts_strogatz_graph(n=n, k=6, p=0.05, seed=seed)

    # Central-favoring: high AQ, weak cues, low initial load
    traits_c, states_c = init_population(n, np.random.default_rng(seed))
    for s in states_c.values():
        s.load = 0.1
    out_c, _ = simulate(g, traits_c, states_c, params,
                        Scenario(messages_per_step=3, aq_range=(0.7, 0.9),
                                 sc_range=(0.2, 0.4), sp_range=(0.2, 0.4),
                                 campaign_end_step=20),
                        n_steps=50, rng=np.random.default_rng(seed))

    # Peripheral-favoring: weaker AQ, strong cues, high initial load
    traits_p, states_p = init_population(n, np.random.default_rng(seed + 100))
    for s in states_p.values():
        s.load = 0.6
    out_p, _ = simulate(g, traits_p, states_p, params,
                        Scenario(messages_per_step=3, aq_range=(0.3, 0.5),
                                 sc_range=(0.7, 0.9), sp_range=(0.7, 0.9),
                                 campaign_end_step=20),
                        n_steps=50, rng=np.random.default_rng(seed + 100))

    df_c = outputs_to_frame(out_c).assign(route="central")
    df_p = outputs_to_frame(out_p).assign(route="peripheral")
    return pd.concat([df_c, df_p], ignore_index=True)


# ── Scenario B: Feasibility-Constrained Behaviour ────────────────────────────

def run_scenario_b(params: Params, seed: int = 42) -> pd.DataFrame:
    """
    Population divided into 4 capability quartiles (Q1=lowest, Q4=highest).
    Three conditions run in parallel from the same random seed:
      - baseline:      no intervention
      - cap_boost:     bottom quartile (Q1) gets cap += 0.2 before simulation
      - fric_reduce:   friction range lowered from (0,1) to (0,0.3) for all agents

    Per-quartile behaviour rates are tracked each step.
    Demonstrates that behaviour uptake is gated by feasibility, not just attitude,
    and that targeted capability or friction interventions have differential effects.
    """
    n = 200
    n_steps = 60
    g = nx.watts_strogatz_graph(n=n, k=6, p=0.05, seed=seed)
    base_rng = np.random.default_rng(seed)

    # Shared initialisation — same population for all three conditions
    traits_base, states_base = init_population(n, base_rng)

    # Assign capability quartiles based on initial cap values
    cap_vals = np.array([states_base[i].cap for i in range(n)])
    quartile_thresholds = np.percentile(cap_vals, [25, 50, 75])
    groups = {}
    for i in range(n):
        c = states_base[i].cap
        if c <= quartile_thresholds[0]:
            groups[i] = "Q1"
        elif c <= quartile_thresholds[1]:
            groups[i] = "Q2"
        elif c <= quartile_thresholds[2]:
            groups[i] = "Q3"
        else:
            groups[i] = "Q4"

    standard_scenario = Scenario(messages_per_step=3)
    low_friction_scenario = Scenario(messages_per_step=3, fr_range=(0.0, 0.3))

    all_dfs = []

    # ── Condition 1: Baseline ────────────────────────────────────────────────
    traits_1 = copy.deepcopy(traits_base)
    states_1 = copy.deepcopy(states_base)
    out_1, _ = simulate(g, traits_1, states_1, params, standard_scenario,
                        n_steps=n_steps, rng=np.random.default_rng(seed + 1),
                        groups=groups)
    df_1 = outputs_to_frame_with_groups(out_1).assign(condition="baseline")
    all_dfs.append(df_1)

    # ── Condition 2: Capability boost for Q1 ────────────────────────────────
    traits_2 = copy.deepcopy(traits_base)
    states_2 = copy.deepcopy(states_base)
    for i, grp in groups.items():
        if grp == "Q1":
            states_2[i].cap = min(1.0, states_2[i].cap + 0.20)
    out_2, _ = simulate(g, traits_2, states_2, params, standard_scenario,
                        n_steps=n_steps, rng=np.random.default_rng(seed + 2),
                        groups=groups)
    df_2 = outputs_to_frame_with_groups(out_2).assign(condition="cap_boost_Q1")
    all_dfs.append(df_2)

    # ── Condition 3: Friction reduction ─────────────────────────────────────
    traits_3 = copy.deepcopy(traits_base)
    states_3 = copy.deepcopy(states_base)
    out_3, _ = simulate(g, traits_3, states_3, params, low_friction_scenario,
                        n_steps=n_steps, rng=np.random.default_rng(seed + 3),
                        groups=groups)
    df_3 = outputs_to_frame_with_groups(out_3).assign(condition="friction_reduce")
    all_dfs.append(df_3)

    return pd.concat(all_dfs, ignore_index=True)


# ── Scenario C: Amplification and Emergent Clustering ─────────────────────────

def run_scenario_c(params: Params, seed: int = 42) -> pd.DataFrame:
    """
    Population split into two identity groups (G0: negative identity_dir preference,
    G1: positive preference). Messages span the full identity_dir range [-1, 1].

    Two conditions:
      - no_amplification: gamma1=0 (visibility independent of engagement)
      - amplification:    gamma1=2.0 (high engagement boosts visibility)

    Tracks attitude variance (polarisation) and exposure_segregation
    (fraction of exposures that are identity-congruent) each step.
    Demonstrates that amplification + identity heterogeneity → opinion clustering
    and filter-bubble exposure patterns.
    """
    n = 200
    n_steps = 60
    g = nx.watts_strogatz_graph(n=n, k=6, p=0.05, seed=seed)

    # Messages span full identity_dir range so both groups see congruent and
    # incongruent content; amplification determines which ones get boosted
    polarised_scenario = Scenario(
        messages_per_step=3,
        identity_dir_range=(-1.0, 1.0),
    )

    all_dfs = []

    for condition, gamma1_val in [("no_amplification", 0.0), ("amplification", 2.0)]:
        p_cond = replace(params, gamma1=gamma1_val)
        traits, states, groups = init_population_identity_groups(
            n, np.random.default_rng(seed)
        )
        out, _ = simulate(
            g, traits, states, p_cond, polarised_scenario,
            n_steps=n_steps,
            rng=np.random.default_rng(seed),
            groups=groups,
            track_segregation=True,
        )
        df = outputs_to_frame_with_groups(out).assign(condition=condition)
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_scenario_b(df: pd.DataFrame, save_path: str = "scenario_b.png"):
    quartiles = ["Q1", "Q2", "Q3", "Q4"]
    conditions = ["baseline", "cap_boost_Q1", "friction_reduce"]
    colors = {"baseline": "steelblue", "cap_boost_Q1": "darkorange", "friction_reduce": "green"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Scenario B: Feasibility-Constrained Behaviour\n"
        "Behaviour rate per capability quartile under three conditions",
        fontsize=12, fontweight="bold"
    )

    for ax, q in zip(axes.flat, quartiles):
        col = f"beh_rate_{q}"
        for cond in conditions:
            sub = df[df["condition"] == cond]
            if col in sub.columns:
                ax.plot(sub["t"], sub[col], label=cond, color=colors[cond])
        ax.set_title(f"Quartile {q}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Behaviour rate")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")


def plot_scenario_c(df: pd.DataFrame, save_path: str = "scenario_c.png"):
    conditions = ["no_amplification", "amplification"]
    colors_cond = {"no_amplification": "steelblue", "amplification": "crimson"}
    colors_group = {"G0": "#1f77b4", "G1": "#ff7f0e"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Scenario C: Amplification and Emergent Clustering\n"
        "Identity groups G0 (negative preference) vs G1 (positive preference)",
        fontsize=12, fontweight="bold"
    )

    # Top-left: mean attitude per group, amplification only
    ax = axes[0, 0]
    sub_amp = df[df["condition"] == "amplification"]
    for grp, color in colors_group.items():
        col = f"mean_att_{grp}"
        if col in sub_amp.columns:
            ax.plot(sub_amp["t"], sub_amp[col], label=grp, color=color)
    ax.set_title("Group attitudes (amplification on)")
    ax.set_xlabel("Step"); ax.set_ylabel("Mean attitude (-1..1)")
    ax.axhline(0, color="grey", lw=0.8, ls="--"); ax.legend()

    # Top-right: mean attitude per group, no amplification
    ax = axes[0, 1]
    sub_noamp = df[df["condition"] == "no_amplification"]
    for grp, color in colors_group.items():
        col = f"mean_att_{grp}"
        if col in sub_noamp.columns:
            ax.plot(sub_noamp["t"], sub_noamp[col], label=grp, color=color)
    ax.set_title("Group attitudes (no amplification)")
    ax.set_xlabel("Step"); ax.set_ylabel("Mean attitude (-1..1)")
    ax.axhline(0, color="grey", lw=0.8, ls="--"); ax.legend()

    # Bottom-left: attitude variance (polarisation) — both conditions
    ax = axes[1, 0]
    for cond, color in colors_cond.items():
        sub = df[df["condition"] == cond]
        ax.plot(sub["t"], sub["att_variance"], label=cond, color=color)
    ax.set_title("Attitude variance (polarisation)")
    ax.set_xlabel("Step"); ax.set_ylabel("Variance"); ax.legend()

    # Bottom-right: exposure segregation — both conditions
    ax = axes[1, 1]
    for cond, color in colors_cond.items():
        sub = df[df["condition"] == cond]
        ax.plot(sub["t"], sub["exposure_segregation"], label=cond, color=color)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", label="baseline (0.5)")
    ax.set_title("Exposure segregation\n(fraction identity-congruent; 0.5 = random)")
    ax.set_xlabel("Step"); ax.set_ylabel("Segregation"); ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/example.yaml")
    ap.add_argument("--out_csv", default="outputs.csv")
    ap.add_argument("--plot", default="outputs.png")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--scenario_a", action="store_true")
    ap.add_argument("--scenario_b", action="store_true")
    ap.add_argument("--scenario_c", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg.get("seed", 42))

    params, scenario, cfg = load_params(cfg)
    g = build_graph(cfg, seed)
    rng = np.random.default_rng(seed)

    traits, states = init_population(cfg["population"]["n_agents"], rng)
    outputs, _ = simulate(g=g, traits=traits, states=states, params=params,
                          scenario=scenario, n_steps=cfg["timing"]["n_steps"], rng=rng)

    df = outputs_to_frame(outputs)
    df.to_csv(args.out_csv, index=False)

    # Main 4-panel plot
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("COM-B + ELM ABM: 200 Agents, 60 Steps", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df["t"], df["mean_att"], color="steelblue")
    ax1.set_title("Mean Attitude"); ax1.axhline(0, color="grey", ls="--", lw=0.8)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df["t"], df["mean_strength"], color="darkorange")
    ax2.set_title("Mean Strength")
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df["t"], df["beh_rate"], color="green")
    ax3.set_title("Behaviour Rate")
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df["t"], df["feasibility_gap"], color="crimson")
    ax4.set_title("Feasibility Gap")
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel("Step")
    plt.savefig(args.plot, dpi=200, bbox_inches="tight")

    print(f"Saved: {args.out_csv}  {args.plot}")
    print(f"Durability half-life: {durability_half_life(df['mean_strength'].to_numpy())}")
    print(f"Final beh_rate: {df['beh_rate'].iloc[-1]:.3f}  |  "
          f"feasibility_gap: {df['feasibility_gap'].iloc[-1]:.3f}  |  "
          f"mean_att: {df['mean_att'].iloc[-1]:.3f}")

    # ── Scenario A ────────────────────────────────────────────────────────────
    if args.scenario_a:
        print("\n── Scenario A: Route-Dependent Durability ──")
        df_a = run_scenario_a(params, seed=seed)
        df_a.to_csv("scenario_a.csv", index=False)
        fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig2.suptitle("Scenario A: Central vs Peripheral Durability (campaign ends step 20)")
        for route, color in [("central", "steelblue"), ("peripheral", "darkorange")]:
            sub = df_a[df_a["route"] == route]
            axes[0].plot(sub["t"], sub["mean_att"], label=route, color=color)
            axes[1].plot(sub["t"], sub["mean_strength"], label=route, color=color)
        for ax, title in zip(axes, ["Mean Attitude", "Mean Strength (durability)"]):
            ax.axvline(20, color="grey", ls="--", lw=1, label="campaign end")
            ax.set_xlabel("Step"); ax.set_title(title); ax.legend()
        plt.tight_layout()
        plt.savefig("scenario_a.png", dpi=200, bbox_inches="tight")
        for route in ["central", "peripheral"]:
            sub = df_a[df_a["route"] == route]["mean_strength"].to_numpy()
            peak = sub.max(); end = sub[-1]
            retention = end / peak if peak > 0 else float("nan")
            print(f"  {route}: peak={peak:.3f}  end={end:.3f}  "
                  f"retention={retention:.1%}  half_life={durability_half_life(sub)}")
        print("Saved: scenario_a.csv  scenario_a.png")

    # ── Scenario B ────────────────────────────────────────────────────────────
    if args.scenario_b:
        print("\n── Scenario B: Feasibility-Constrained Behaviour ──")
        df_b = run_scenario_b(params, seed=seed)
        df_b.to_csv("scenario_b.csv", index=False)
        plot_scenario_b(df_b, "scenario_b.png")
        # Summary table: final-step beh_rate per quartile per condition
        print("\n  Final-step behaviour rates by quartile and condition:")
        final = df_b[df_b["t"] == df_b["t"].max()]
        for cond in ["baseline", "cap_boost_Q1", "friction_reduce"]:
            row = final[final["condition"] == cond].iloc[0]
            rates = {q: f"{row[f'beh_rate_{q}']:.2f}" for q in ["Q1","Q2","Q3","Q4"]
                     if f"beh_rate_{q}" in row}
            print(f"  {cond:20s}: {rates}")
        print("Saved: scenario_b.csv  scenario_b.png")

    # ── Scenario C ────────────────────────────────────────────────────────────
    if args.scenario_c:
        print("\n── Scenario C: Amplification and Emergent Clustering ──")
        df_c = run_scenario_c(params, seed=seed)
        df_c.to_csv("scenario_c.csv", index=False)
        plot_scenario_c(df_c, "scenario_c.png")
        # Summary: final att variance and mean segregation per condition
        print("\n  Polarisation and segregation summary:")
        for cond in ["no_amplification", "amplification"]:
            sub = df_c[df_c["condition"] == cond]
            final_var = sub["att_variance"].iloc[-1]
            mean_seg = sub["exposure_segregation"].mean()
            g0_att = sub["mean_att_G0"].iloc[-1]
            g1_att = sub["mean_att_G1"].iloc[-1]
            print(f"  {cond:20s}: att_variance={final_var:.4f}  "
                  f"mean_segregation={mean_seg:.3f}  "
                  f"G0_att={g0_att:.3f}  G1_att={g1_att:.3f}")
        print("Saved: scenario_c.csv  scenario_c.png")

    # ── Verification ──────────────────────────────────────────────────────────
    if args.verify:
        print("\n── Verification: route monotonicity ──")
        print(verify_route_monotonicity(params, seed=seed))
        print("── Ablation: visibility gamma1=0 ──")
        print(ablation_visibility(params, seed=seed))


if __name__ == "__main__":
    main()
