import pandas as pd

def portfolio_summary(customers: pd.DataFrame):
    n=len(customers)
    positive=int((customers.uplift>0).sum())
    targetable=int((customers.recommended_action=="Target").sum())
    return {
        "customers": n,
        "positive_uplift_customers": positive,
        "positive_uplift_rate": positive/n if n else 0,
        "target_recommended": targetable,
        "target_rate": targetable/n if n else 0,
        "mean_uplift": float(customers.uplift.mean()) if n else 0,
        "median_uplift": float(customers.uplift.median()) if n else 0,
        "max_uplift": float(customers.uplift.max()) if n else 0,
        "segments": customers.segment_label.value_counts().to_dict(),
    }
