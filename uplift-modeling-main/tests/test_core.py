from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient
from backend.main import app
from src.data.loader import load_hillstrom
from src.evaluation.metrics import qini_curve, auuc
from src.optimization.business import optimize_budget
ROOT=Path(__file__).resolve().parents[1]

def test_data_schema():
    df=load_hillstrom(ROOT/"data/hillstrom.csv"); assert df.shape[0]==64000; assert df["conversion"].isin([0,1]).all()
def test_qini_monotonic_x():
    c=qini_curve(np.array([0,1,0,1]),np.array([0,1,0,1]),np.array([.1,.9,.2,.8])); assert np.all(np.diff(c.fraction)>0); assert np.isfinite(auuc(c))
def test_budget_validation():
    import pandas as pd
    d=pd.DataFrame({"customer_id":[1,2],"uplift":[.1,-.1]}); r=optimize_budget(d,10,1,20); assert r["targeted_count"]==1
def test_api_health(): assert TestClient(app).get('/health').status_code==200
def test_uplift_budget_is_bounded():
    import pandas as pd
    d=pd.DataFrame({'customer_id':[1,2],'uplift':[1.0,-1.0]}); r=optimize_budget(d,10,1,20); assert r['targeted_count']==1 and r['expected_incremental_conversions']==1
