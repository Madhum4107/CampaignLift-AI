import pandas as pd
from src.optimization.business import assign_segments

def test_segments_exist():
    d=pd.DataFrame({"uplift":[.2,-.1,.1],"p_control":[.01,.5,.8]}); out=assign_segments(d); assert set(out.segment_label).issubset({"Persuadables","Sure Things","Sleeping Dogs","Lost Causes"})
