import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

def _comparison(df: pd.DataFrame, treatment_label: str, outcome: str = "conversion") -> dict:
    t = df[df["segment"] == treatment_label][outcome]
    c = df[df["segment"] == "No E-Mail"][outcome]
    rate_t, rate_c = float(t.mean()), float(c.mean())
    diff = rate_t - rate_c
    rel = diff / rate_c if rate_c else 0.0
    stat, p = proportions_ztest([int(t.sum()), int(c.sum())], [len(t), len(c)])
    ci_t = proportion_confint(int(t.sum()), len(t), alpha=0.05, method="wilson")
    ci_c = proportion_confint(int(c.sum()), len(c), alpha=0.05, method="wilson")
    return {"treatment": treatment_label, "treatment_n": len(t), "control_n": len(c), "treatment_rate": rate_t, "control_rate": rate_c, "absolute_lift": diff, "relative_lift": rel, "z_stat": float(stat), "p_value": float(p), "treatment_ci": [float(ci_t[0]), float(ci_t[1])], "control_ci": [float(ci_c[0]), float(ci_c[1])]}

def experiment_stats(df: pd.DataFrame, outcome: str = "conversion") -> dict:
    results = [_comparison(df, "Mens E-Mail", outcome), _comparison(df, "Womens E-Mail", outcome)]
    return {"outcome": outcome, "results": results}
