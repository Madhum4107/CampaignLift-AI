from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
import io
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas.models import BudgetRequest, PolicyRequest, GuardrailRequest
from src.data.loader import load_hillstrom
from src.preprocessing.features import feature_frame, FEATURES
from src.experimentation.stats import experiment_stats
from src.models.pipeline import load_artifacts
from src.optimization.business import optimize_budget, simulate_policies, apply_guardrails, campaign_readiness
from src.monitoring.summary import portfolio_summary

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data"/"hillstrom.csv"; MODELS=ROOT/"models"
app=FastAPI(title="CampaignLift AI API",version="2.2.0",description="Uplift modeling and campaign targeting API")
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173","http://localhost:5174","http://localhost:5175","http://localhost:4173"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
_bundle=_customers=_metrics=None

def artifacts():
    global _bundle,_customers,_metrics
    if _bundle is None:
        try: _bundle,_customers,_metrics=load_artifacts(MODELS)
        except FileNotFoundError as e: raise HTTPException(503,"Model artifacts are missing. Run python train.py first.") from e
    return _bundle,_customers,_metrics

def invalidate_artifacts_cache():
    global _bundle,_customers,_metrics; _bundle=_customers=_metrics=None

@app.get("/",tags=["system"])
def root(): return {"service":"CampaignLift AI API","version":app.version,"docs":"/docs","health":"/health"}
@app.get("/health",tags=["system"])
def health(): return {"status":"ok","service":"campaignlift-ai","version":app.version}
@app.get("/summary",tags=["analytics"])
def summary():
    _,df,_=artifacts(); return portfolio_summary(df)
@app.get("/experiment/stats",tags=["analytics"])
def stats(): return experiment_stats(load_hillstrom(DATA),"conversion")
@app.get("/model/metrics",tags=["analytics"])
def metrics(): _,_,m=artifacts(); return m

@app.get("/segments",tags=["analytics"])
def segments():
    _,df,_=artifacts()
    rows=[]
    for label, g in df.groupby("segment_label", sort=False):
        rows.append({
            "segment": label,
            "customers": int(len(g)),
            "share": float(len(g)/len(df)) if len(df) else 0.0,
            "avg_uplift": float(g.uplift.mean()) if len(g) else 0.0,
            "avg_control_probability": float(g.p_control.mean()) if len(g) else 0.0,
            "recommended_action": str(g.recommended_action.mode().iloc[0]) if len(g) else "",
        })
    order={"Persuadables":0,"Sure Things":1,"Sleeping Dogs":2,"Lost Causes":3}
    rows.sort(key=lambda x: order.get(x["segment"],99))
    return {"segments":rows}
@app.get("/customers",tags=["customers"])
def customers(limit:int=Query(50,ge=1,le=500),offset:int=Query(0,ge=0),segment:str|None=None,min_uplift:float|None=None,max_uplift:float|None=None):
    _,df,_=artifacts(); out=df
    if segment: out=out[out.segment_label==segment]
    if min_uplift is not None: out=out[out.uplift>=min_uplift]
    if max_uplift is not None: out=out[out.uplift<=max_uplift]
    filtered_count=len(out)
    out=out.sort_values("uplift",ascending=False).iloc[offset:offset+limit]
    return {"total":len(df),"filtered":filtered_count,"offset":offset,"limit":limit,"items":out.to_dict("records")}
@app.get("/customers/export",tags=["customers"])
def export_customers(segment:str|None=None,min_uplift:float|None=None,max_uplift:float|None=None,limit:int=Query(1000,ge=1,le=64000)):
    _,df,_=artifacts(); out=df.copy()
    if segment: out=out[out.segment_label==segment]
    if min_uplift is not None: out=out[out.uplift>=min_uplift]
    if max_uplift is not None: out=out[out.uplift<=max_uplift]
    out=out.sort_values("uplift",ascending=False).head(limit)
    cols=[c for c in ["customer_id","segment_label","uplift","p_control","recommended_action","recency","history","channel","segment"] if c in out.columns]
    buf=io.StringIO(); out[cols].to_csv(buf,index=False)
    return StreamingResponse(iter([buf.getvalue()]),media_type="text/csv",headers={"Content-Disposition":"attachment; filename=campaignlift_targets.csv"})

@app.get("/customers/{customer_id}",tags=["customers"])
def customer(customer_id:int):
    bundle,df,_=artifacts(); row=df[df.customer_id==customer_id]
    if row.empty: raise HTTPException(404,"Customer not found")
    r=row.iloc[0].to_dict(); raw=load_hillstrom(DATA); rawrow=raw.iloc[[customer_id]].copy() if 0<=customer_id<len(raw) else None; drivers=[]
    if rawrow is not None:
        Xbase=bundle["preprocessor"].transform(feature_frame(rawrow)); model=bundle["model"]; base=float(model.predict_uplift(Xbase)[0])
        for feature in FEATURES:
            pert=rawrow.copy(); pert[feature]=raw[feature].mode().iloc[0] if feature in ["history_segment","zip_code","channel"] else raw[feature].median()
            changed=float(model.predict_uplift(bundle["preprocessor"].transform(feature_frame(pert)))[0]); drivers.append({"feature":feature,"local_effect":base-changed})
        drivers=sorted(drivers,key=lambda x:abs(x["local_effect"]),reverse=True)[:5]
    profile={}
    if rawrow is not None:
        profile={k:(v.item() if hasattr(v,"item") else v) for k,v in rawrow.iloc[0].to_dict().items()}
    return {"customer_id":customer_id,"uplift_score":float(r["uplift"]),"control_purchase_probability":float(r["p_control"]),"segment":r["segment_label"],"recommended_action":r["recommended_action"],"profile":profile,"drivers":drivers,"explanation_note":"Local sensitivity changes one feature to a population reference value. It describes model sensitivity, not causal feature influence."}
@app.post("/optimize/budget",tags=["optimization"])
def budget(req:BudgetRequest):
    _,df,_=artifacts(); return optimize_budget(df,req.budget,req.cost_per_contact,req.revenue_per_incremental_conversion)
@app.post("/simulate/policy",tags=["optimization"])
def policy(req:PolicyRequest):
    _,df,_=artifacts(); return {"policies":simulate_policies(df,req.revenue_per_incremental_conversion,req.cost_per_contact)}

@app.post("/optimize/budget/sweep",tags=["optimization"])
def budget_sweep(req:BudgetRequest):
    _,df,_=artifacts()
    rows=[]
    for share in (0.25,0.50,0.75,1.00):
        b=req.budget*share
        result=optimize_budget(df,b,req.cost_per_contact,req.revenue_per_incremental_conversion)
        rows.append({"budget":b,"targeted_count":result["targeted_count"],"expected_incremental_conversions":result["expected_incremental_conversions"],"expected_incremental_revenue":result["expected_incremental_revenue"],"expected_profit":result["expected_profit"],"roi":result["roi"]})
    return {"scenarios":rows}



@app.post("/campaign/guardrails",tags=["optimization"])
def campaign_guardrails(req:GuardrailRequest):
    _,df,_=artifacts()
    return apply_guardrails(df,req.budget,req.cost_per_contact,req.revenue_per_incremental_conversion,req.min_uplift,req.max_contacts,req.exclude_segments)

@app.get("/campaign/readiness",tags=["monitoring"])
def campaign_readiness_check():
    _,df,m=artifacts(); return campaign_readiness(df,m)

@app.get("/monitoring/health",tags=["monitoring"])
def monitoring_health():
    _,df,m=artifacts()
    numeric=df.select_dtypes(include="number")
    missing={k:int(v) for k,v in df.isna().sum().items() if int(v)>0}
    return {"rows":len(df),"columns":len(df.columns),"missing_values":missing,"duplicate_customer_ids":int(df.customer_id.duplicated().sum()) if "customer_id" in df.columns else None,"uplift_mean":float(df.uplift.mean()),"uplift_std":float(df.uplift.std()),"uplift_positive_rate":float((df.uplift>0).mean()),"model_best":m.get("best_model"),"numeric_columns":len(numeric.columns),"status":"healthy" if not missing and (df.customer_id.nunique()==len(df) if "customer_id" in df.columns else True) else "review"}
