from fastapi.testclient import TestClient
from backend.main import app
client=TestClient(app)

def test_root():
    r=client.get('/'); assert r.status_code==200; assert r.json()['service']=='CampaignLift AI API'
def test_summary():
    r=client.get('/summary'); assert r.status_code==200; assert r.json()['customers']==64000
def test_model_metrics():
    r=client.get('/model/metrics'); assert r.status_code==200; assert r.json()['best_model'] in {'S-Learner','T-Learner','X-Learner'}
def test_customers_filter():
    r=client.get('/customers?limit=5&min_uplift=0'); assert r.status_code==200; assert len(r.json()['items'])<=5
def test_missing_customer():
    r=client.get('/customers/999999'); assert r.status_code==404
def test_policy():
    r=client.post('/simulate/policy',json={'revenue_per_incremental_conversion':50,'cost_per_contact':0.05}); assert r.status_code==200; assert len(r.json()['policies'])==5

def test_segments():
    r=client.get('/segments'); assert r.status_code==200; assert len(r.json()['segments'])==4

def test_budget_sweep():
    r=client.post('/optimize/budget/sweep',json={'budget':500,'cost_per_contact':0.05,'revenue_per_incremental_conversion':50}); assert r.status_code==200; assert len(r.json()['scenarios'])==4

def test_customer_export():
    r=client.get('/customers/export?min_uplift=0&limit=5'); assert r.status_code==200; assert r.headers['content-type'].startswith('text/csv'); assert 'customer_id' in r.text


def test_campaign_guardrails():
    r=client.post('/campaign/guardrails',json={'budget':500,'cost_per_contact':0.05,'revenue_per_incremental_conversion':50,'min_uplift':0.01,'max_contacts':1000,'exclude_segments':['Sleeping Dogs']})
    assert r.status_code==200; assert r.json()['targeted_count']<=1000

def test_campaign_readiness():
    r=client.get('/campaign/readiness'); assert r.status_code==200; assert 'checks' in r.json()

def test_monitoring_health():
    r=client.get('/monitoring/health'); assert r.status_code==200; assert r.json()['rows']==64000
