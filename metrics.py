from __future__ import annotations
from typing import List, Optional
import pandas as pd
import numpy as np
from .sim import StepOutputs


def outputs_to_frame(outputs: List[StepOutputs], group_label: Optional[str] = None) -> pd.DataFrame:
    """
    Convert simulation outputs to a DataFrame.
    If group_label is provided, includes that group's beh_rate and mean_att
    as additional columns (used by Scenario B and C).
    """
    rows = []
    for o in outputs:
        row = {
            "t": len(rows),
            "mean_att": o.mean_att,
            "mean_strength": o.mean_strength,
            "mean_intent": o.mean_intent,
            "beh_rate": o.beh_rate,
            "feasibility_gap": o.feasibility_gap,
            "att_variance": o.att_variance,
            "exposure_segregation": o.exposure_segregation,
        }
        if group_label is not None:
            row[f"beh_rate_{group_label}"] = o.group_beh_rates.get(group_label, float("nan"))
            row[f"mean_att_{group_label}"] = o.group_mean_atts.get(group_label, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def outputs_to_frame_with_groups(outputs: List[StepOutputs]) -> pd.DataFrame:
    """
    Expand all group_beh_rates and group_mean_atts into columns.
    Used when you want one wide DataFrame with all group columns.
    """
    base = outputs_to_frame(outputs)
    # Collect all group labels across all steps
    all_labels = set()
    for o in outputs:
        all_labels.update(o.group_beh_rates.keys())
        all_labels.update(o.group_mean_atts.keys())
    for lbl in sorted(all_labels):
        base[f"beh_rate_{lbl}"] = [o.group_beh_rates.get(lbl, float("nan")) for o in outputs]
        base[f"mean_att_{lbl}"] = [o.group_mean_atts.get(lbl, float("nan")) for o in outputs]
    return base


def durability_half_life(series: np.ndarray) -> float:
    """Half-life after peak: steps to drop to 50% of peak value."""
    if len(series) < 3:
        return float("nan")
    peak_idx = int(series.argmax())
    peak = float(series[peak_idx])
    if peak <= 0:
        return float("nan")
    target = 0.5 * peak
    for k in range(peak_idx + 1, len(series)):
        if series[k] <= target:
            return float(k - peak_idx)
    return float("nan")


def polarisation_index(att_series: np.ndarray) -> float:
    """
    Simple polarisation index: variance of the attitude distribution over time.
    Higher variance = more spread between agents (Scenario C metric).
    Returns the mean variance across all steps.
    """
    return float(np.nanmean(att_series))
