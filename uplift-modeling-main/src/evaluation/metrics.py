import numpy as np
import pandas as pd

def qini_curve(y, treatment, uplift):
    y=np.asarray(y); t=np.asarray(treatment); u=np.asarray(uplift)
    if not (len(y)==len(t)==len(u)) or len(y)==0:
        raise ValueError("y, treatment and uplift must have the same non-zero length")
    if not np.isfinite(u).all(): raise ValueError("uplift contains non-finite values")
    order=np.argsort(-u, kind="mergesort"); y=y[order]; t=t[order]
    n=len(y); nt=t.sum(); nc=n-nt
    if nt==0 or nc==0: raise ValueError("Both treatment and control groups are required")
    ratio=nt/nc
    cum_t=np.cumsum(y*t); cum_c=np.cumsum(y*(1-t))
    inc=cum_t-ratio*cum_c
    x=np.arange(1,n+1)/n
    return pd.DataFrame({"fraction":x,"incremental":inc})

def auuc(curve: pd.DataFrame) -> float:
    return float(np.trapezoid(curve["incremental"], curve["fraction"]))

def qini_coefficient(curve: pd.DataFrame) -> float:
    total=float(curve["incremental"].iloc[-1]); return auuc(curve)-0.5*total

def uplift_at_k(y,treatment,uplift,k):
    if not 0 < k <= 1: raise ValueError("k must be in (0, 1]")
    c=qini_curve(y,treatment,uplift); idx=max(1,int(len(c)*k))-1; return float(c.iloc[idx]["incremental"])

def evaluate_uplift(y,treatment,uplift):
    c=qini_curve(y,treatment,uplift)
    return {
        "auuc":auuc(c),
        "qini_coefficient":qini_coefficient(c),
        "uplift_at_10":uplift_at_k(y,treatment,uplift,.10),
        "uplift_at_20":uplift_at_k(y,treatment,uplift,.20),
        "uplift_at_30":uplift_at_k(y,treatment,uplift,.30),
        "curve":c,
    }
