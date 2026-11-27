import pandas as pd
import pytest
from src.optimization.business import optimize_budget, simulate_policies, apply_guardrails, assign_segments, campaign_readiness

@pytest.fixture
def customers():
    return pd.DataFrame({
        "customer_id":[10,20,30,40],
        "uplift":[0.30,0.15,0.02,-0.10],
        "p_control":[0.10,0.70,0.20,0.50],
        "segment_label":["Persuadables","Sure Things","Persuadables","Sleeping Dogs"],
        "recommended_action":["Target","Avoid paid contact","Target","Do not target"],
    })

def test_segments_are_assigned_without_mutating_source():
    source=pd.DataFrame({"uplift":[.2,-.1,.1],"p_control":[.01,.5,.8]})
    out=assign_segments(source)
    assert set(out.segment_label).issubset({"Persuadables","Sure Things","Sleeping Dogs","Lost Causes"})
    assert "segment_label" not in source.columns

def test_budget_optimizer_enforces_economic_and_budget_constraints(customers):
    result=optimize_budget(customers,budget=1.0,cost_per_contact=.5,revenue_per_incremental_conversion=2)
    assert result["targeted_count"] <= 2
    assert result["total_cost"] <= 1.0
    assert result["selected_customers"] == [10,20]
    assert result["expected_profit"] == pytest.approx(.4)

def test_zero_contact_cost_and_zero_budget_are_safe(customers):
    free=optimize_budget(customers,10,0,2)
    assert free["targeted_count"]==3
    empty=optimize_budget(customers,0,.5,2)
    assert empty["targeted_count"]==0

@pytest.mark.parametrize("inputs", [(-1,1,1),(1,-1,1),(1,1,-1)])
def test_budget_optimizer_rejects_negative_values(customers,inputs):
    with pytest.raises(ValueError):
        optimize_budget(customers,*inputs)

def test_guardrails_filter_segments_min_uplift_and_max_contacts(customers):
    result=apply_guardrails(customers,2,.5,2,min_uplift=.1,max_contacts=1,exclude_segments=["Sure Things"])
    assert result["selected_customers"]==[10]
    assert result["total_cost"]==.5
    assert result["eligible_after_guardrails"]==1

def test_guardrails_return_empty_if_budget_cannot_contact(customers):
    result=apply_guardrails(customers,0,.5,10)
    assert result["targeted_count"]==0

def test_policy_simulation_returns_expected_policy_set(customers):
    result=simulate_policies(customers,10,.1)
    assert [x["policy"] for x in result]==["Random","Top 5%","Top 10%","Top 20%","Top 30%"]
    assert all(x["targeted_count"]>=1 for x in result)

def test_readiness_reports_not_ready_for_no_targetable_customers():
    frame=pd.DataFrame({"uplift":[-.1,-.2],"recommended_action":["Avoid","Avoid"]})
    result=campaign_readiness(frame,{})
    assert result["overall"]=="not_ready"
