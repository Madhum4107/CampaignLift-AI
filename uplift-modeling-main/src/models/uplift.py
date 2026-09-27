from dataclasses import dataclass
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

@dataclass
class ModelConfig:
    n_estimators: int = 120
    max_depth: int = 8
    random_state: int = 42

class BaseUplift:
    name = "base"
    def fit(self, X, treatment, y): raise NotImplementedError
    def predict_uplift(self, X): raise NotImplementedError

    @staticmethod
    def _clip(uplift):
        # Individual treatment effect for a binary outcome is bounded to [-1, 1].
        return np.clip(np.asarray(uplift, dtype=float), -1.0, 1.0)

class SLearner(BaseUplift):
    name = "S-Learner"
    def __init__(self, estimator=None):
        self.estimator = estimator or LogisticRegression(max_iter=1000)
    def fit(self, X, treatment, y):
        self.model = clone(self.estimator)
        self.model.fit(np.column_stack([X, treatment]), y)
        return self
    def predict_uplift(self, X):
        n=len(X); x0=np.column_stack([X, np.zeros(n)]); x1=np.column_stack([X, np.ones(n)])
        return self._clip(self.model.predict_proba(x1)[:,1]-self.model.predict_proba(x0)[:,1])

class TLearner(BaseUplift):
    name = "T-Learner"
    def __init__(self, estimator=None):
        self.estimator = estimator or RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1)
    def fit(self, X, treatment, y):
        self.control = clone(self.estimator); self.treated = clone(self.estimator)
        self.control.fit(X[treatment==0], y[treatment==0]); self.treated.fit(X[treatment==1], y[treatment==1]); return self
    def predict_uplift(self, X):
        return self._clip(self.treated.predict_proba(X)[:,1]-self.control.predict_proba(X)[:,1])

class XLearner(BaseUplift):
    name = "X-Learner"
    def __init__(self, outcome_estimator=None, effect_estimator=None):
        self.outcome_estimator = outcome_estimator or RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1)
        self.effect_estimator = effect_estimator or RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1)
    def fit(self, X, treatment, y):
        m0=clone(self.outcome_estimator); m1=clone(self.outcome_estimator)
        m0.fit(X[treatment==0],y[treatment==0]); m1.fit(X[treatment==1],y[treatment==1])
        mu0=m0.predict_proba(X)[:,1]; mu1=m1.predict_proba(X)[:,1]
        d1=y[treatment==1]-mu0[treatment==1]; d0=mu1[treatment==0]-y[treatment==0]
        e1=clone(self.effect_estimator); e0=clone(self.effect_estimator)
        e1.fit(X[treatment==1],d1); e0.fit(X[treatment==0],d0)
        self.m0,self.m1,self.e0,self.e1=m0,m1,e0,e1
        self.w=float(treatment.mean()); return self
    def predict_uplift(self,X):
        return self._clip((1-self.w)*self.e1.predict(X)+self.w*self.e0.predict(X))

def default_models(): return [SLearner(), TLearner(), XLearner()]
