from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from src.data.loader import load_hillstrom, add_binary_treatment
from src.preprocessing.features import build_preprocessor, feature_frame
from src.models.uplift import SLearner,TLearner,XLearner
from src.evaluation.metrics import evaluate_uplift

class UpliftPipeline:
    def __init__(self,data_path="data/hillstrom.csv",artifact_dir="models"):
        self.data_path=Path(data_path); self.artifact_dir=Path(artifact_dir); self.artifact_dir.mkdir(parents=True,exist_ok=True)
    def fit(self):
        df=add_binary_treatment(load_hillstrom(self.data_path)); y=df.conversion.to_numpy(); t=df.treatment.to_numpy(); Xraw=feature_frame(df)
        idx=np.arange(len(df)); strat=(t.astype(str)+y.astype(str)); tr, temp=train_test_split(idx,test_size=.30,random_state=42,stratify=strat); va,te=train_test_split(temp,test_size=.50,random_state=42,stratify=strat[temp])
        pre=build_preprocessor(); X=pre.fit_transform(Xraw); Xv=X[va]; Xt=X[te]
        models=[SLearner(),TLearner(),XLearner()]; results={}; fitted={}
        for m in models:
            m.fit(X[tr],t[tr],y[tr]); ev=evaluate_uplift(y[va],t[va],m.predict_uplift(Xv)); results[m.name]={k:v for k,v in ev.items() if k!="curve"}; fitted[m.name]=m
        best=max(results,key=lambda n:results[n]["qini_coefficient"])
        final=type(fitted[best])(); final.fit(X[np.concatenate([tr,va])],t[np.concatenate([tr,va])],y[np.concatenate([tr,va])]); test_ev=evaluate_uplift(y[te],t[te],final.predict_uplift(Xt)); results[best].update({"test_auuc":test_ev["auuc"],"test_qini_coefficient":test_ev["qini_coefficient"]})
        all_up=final.predict_uplift(X)
        if best=="T-Learner": p0=final.control.predict_proba(X)[:,1]
        elif best=="X-Learner": p0=final.m0.predict_proba(X)[:,1]
        else: p0=final.model.predict_proba(np.column_stack([X,np.zeros(len(X))]))[:,1]
        customers=df[["recency","history","channel","segment"]].copy(); customers.insert(0,"customer_id",np.arange(len(df))); customers["uplift"]=all_up; customers["p_control"]=p0
        from src.optimization.business import assign_segments
        customers=assign_segments(customers); customers.to_csv(self.artifact_dir/"customer_scores.csv",index=False)
        joblib.dump({"preprocessor":pre,"model":final,"best_model":best},self.artifact_dir/"uplift_model.joblib")
        curves={}
        for name,m in fitted.items():
            c=evaluate_uplift(y[va],t[va],m.predict_uplift(Xv))["curve"]; curves[name]=c.iloc[::max(1,len(c)//100)].to_dict("records")
        with open(self.artifact_dir/"metrics.json","w") as f: json.dump({"models":results,"best_model":best,"curves":curves,"training":{"rows":len(df),"train_rows":len(tr),"validation_rows":len(va),"test_rows":len(te),"treatment_rate":float(t.mean()),"seed":42}},f,indent=2)
        return results,best

def load_artifacts(artifact_dir="models"):
    d=Path(artifact_dir); bundle=joblib.load(d/"uplift_model.joblib"); customers=pd.read_csv(d/"customer_scores.csv"); metrics=json.loads((d/"metrics.json").read_text()); return bundle,customers,metrics
