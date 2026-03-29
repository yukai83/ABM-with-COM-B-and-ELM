from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List


def outputs_to_frame(outputs) -> pd.DataFrame:
    rows = []
    for t, o in enumerate(outputs):
        rows.append({
            "t": t,
            "mean_att": o.mean_att,
            "mean_strength": o.mean_strength,
            "mean_intent": o.mean_intent,
            "beh_rate": o.beh_rate,
            "feasibility_gap": o.feasibility_gap,
            "att_variance": o.att_variance,
            "exposure_segregation": o.exposure_segregation,
        })
    return pd.DataFrame(rows)


def outputs_to_frame_with_groups(outputs) -> pd.DataFrame:
    rows = []
    for t, o in enumerate(outputs):
        row = {
            "t": t,
            "mean_att": o.mean_att,
            "mean_strength": o.mean_strength,
            "mean_intent": o.mean_intent,
            "beh_rate": o.beh_rate,
            "feasibility_gap": o.feasibility_gap,
            "att_variance": o.att_variance,
            "exposure_segregation": o.exposure_segregation,
        }
        for lbl, val in o.group_beh_rates.items():
            row[f"beh_rate_{lbl}"] = val
        for lbl, val in o.group_mean_atts.items():
            row[f"mean_att_{lbl}"] = val
        rows.append(row)
    return pd.DataFrame(rows)


def durability_half_life(strength_series: np.ndarray) -> float:
    """Steps after peak for mean strength to halve. Returns nan if never halves."""
    peak_idx = int(np.argmax(strength_series))
    peak_val = strength_series[peak_idx]
    half_val = peak_val / 2.0
    for i in range(peak_idx, len(strength_series)):
        if strength_series[i] <= half_val:
            return float(i - peak_idx)
    return float("nan")
