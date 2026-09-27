import numpy as np
import pandas as pd

def assign_segments(df, uplift_col="uplift", p_control_col="p_control"):
    out=df.copy()
    median=float(out[p_control_col].median())
    def seg(r):
        if r[uplift_col] < 0: return "Sleeping Dogs"
        if r[p_control_col] >= median: return "Sure Things"
        return "Persuadables"
    out["segment_label"]=[seg(r) for _,r in out.iterrows()]
    out.loc[(out[uplift_col] <= 0) & (out[p_control_col] < median),"segment_label"]="Lost Causes"
    out["recommended_action"]=out["segment_label"].map({"Persuadables":"Target","Sure Things":"Avoid paid contact","Sleeping Dogs":"Do not target","Lost Causes":"Do not target"})
    return out

def optimize_budget(customers,budget,cost_per_contact,revenue_per_incremental_conversion):
    if min(budget,cost_per_contact,revenue_per_incremental_conversion) < 0: raise ValueError("Inputs cannot be negative")
    if cost_per_contact == 0: capacity=len(customers)
    else: capacity=min(len(customers),int(budget//cost_per_contact))
    ranked=customers.sort_values("uplift",ascending=False).copy()
    ranked["expected_incremental_revenue_per_contact"]=ranked["uplift"]*revenue_per_incremental_conversion
    ranked=ranked[(ranked["uplift"] > 0) & (ranked["expected_incremental_revenue_per_contact"]>=cost_per_contact)].head(capacity)
    expected_conv=float(ranked["uplift"].sum()); cost=float(len(ranked)*cost_per_contact); revenue=expected_conv*revenue_per_incremental_conversion; profit=revenue-cost
    return {"selected_customers":ranked["customer_id"].tolist(),"targeted_count":len(ranked),"total_cost":cost,"expected_incremental_conversions":expected_conv,"expected_incremental_revenue":revenue,"expected_profit":profit,"roi":(profit/cost if cost else 0.0)}

def simulate_policies(customers,revenue_per_incremental_conversion=50,cost_per_contact=.05):
    if min(revenue_per_incremental_conversion,cost_per_contact) < 0: raise ValueError("Inputs cannot be negative")
    rows=[]; ranked=customers.sort_values("uplift",ascending=False).reset_index(drop=True)
    for label,p in [("Random",1.0),("Top 5%",.05),("Top 10%",.10),("Top 20%",.20),("Top 30%",.30)]:
        n=len(ranked) if p==1 else max(1,int(len(ranked)*p))
        inc=float(ranked["uplift"].clip(lower=0).mean()*n) if label == "Random" else float(ranked.head(n)["uplift"].clip(lower=0).sum())
        cost=n*cost_per_contact; rev=inc*revenue_per_incremental_conversion; profit=rev-cost
        rows.append({"policy":label,"targeted_count":n,"expected_incremental_conversions":inc,"cost":cost,"revenue":rev,"profit":profit,"roi":profit/cost if cost else 0})
    return rows


def apply_guardrails(customers, budget, cost_per_contact, revenue_per_incremental_conversion,
                    min_uplift=0.0, max_contacts=None, exclude_segments=None):
    if min(budget, cost_per_contact, revenue_per_incremental_conversion) < 0:
        raise ValueError("Inputs cannot be negative")
    exclude_segments=set(exclude_segments or [])
    out=customers.copy()
    before=len(out)
    if "segment_label" in out.columns and exclude_segments:
        out=out[~out.segment_label.isin(exclude_segments)]
    out=out[out.uplift >= min_uplift]
    if cost_per_contact > 0:
        capacity=int(budget//cost_per_contact)
    else:
        capacity=len(out)
    if max_contacts is not None:
        capacity=min(capacity,max_contacts)
    ranked=out.sort_values("uplift",ascending=False).head(capacity).copy()
    ranked["expected_incremental_revenue"]=ranked.uplift*revenue_per_incremental_conversion
    ranked["contact_cost"]=cost_per_contact
    ranked["expected_incremental_profit"]=ranked.expected_incremental_revenue-cost_per_contact
    selected_cost=float(len(ranked)*cost_per_contact)
    expected_conv=float(ranked.uplift.sum())
    revenue=float(expected_conv*revenue_per_incremental_conversion)
    profit=float(revenue-selected_cost)
    return {
        "eligible_before_guardrails": before,
        "eligible_after_guardrails": len(out),
        "selected_customers": ranked.customer_id.tolist(),
        "targeted_count": len(ranked),
        "total_cost": selected_cost,
        "expected_incremental_conversions": expected_conv,
        "expected_incremental_revenue": revenue,
        "expected_profit": profit,
        "roi": profit/selected_cost if selected_cost else 0.0,
        "constraints": {"min_uplift": min_uplift, "max_contacts": max_contacts, "exclude_segments": sorted(exclude_segments)}
    }

def campaign_readiness(customers, metrics):
    n=len(customers)
    positive=float((customers.uplift>0).mean()) if n else 0
    targetable=float((customers.recommended_action=="Target").mean()) if n else 0
    x=metrics.get("models",{}).get("X-Learner",{})
    checks=[
        {"name":"Model artifact available","status":"pass" if x else "warn","detail":"Selected-model metrics are available." if x else "No X-Learner metrics found."},
        {"name":"Positive uplift coverage","status":"pass" if positive>=0.10 else "warn","detail":f"{positive:.1%} of customers have positive estimated uplift."},
        {"name":"Targetable population","status":"pass" if targetable>0 else "fail","detail":f"{targetable:.1%} are currently recommended for targeting."},
        {"name":"Holdout evaluation","status":"pass" if x.get("test_qini_coefficient") is not None else "warn","detail":f"Test Qini = {x.get('test_qini_coefficient','n/a')} for X-Learner."},
    ]
    return {"overall":"ready_with_cautions" if all(c["status"]!="fail" for c in checks) else "not_ready", "checks":checks, "disclaimer":"This is a portfolio decision-support check, not a production causal guarantee. Fresh randomized experimentation is required before deployment."}
