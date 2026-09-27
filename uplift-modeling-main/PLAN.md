# CampaignLift AI — Final Implementation Plan

## Completed

1. Validate Hillstrom data and treatment assignment.
2. Run randomized experiment analysis for Mens and Womens Email against control.
3. Train S-, T-, and X-Learners using train/validation/test separation.
4. Select the model using validation Qini coefficient and report final isolated test metrics.
5. Keep outcome probabilities calibrated for the rare conversion target by avoiding class-rebalancing weights in the outcome models.
6. Bound individual binary-outcome treatment effects to `[-1, 1]`.
7. Score all customers and assign business segments.
8. Add customer-level model-sensitivity explanations.
9. Expose summary, experiment metrics, model metrics, customer ranking/filtering, customer detail, budget optimization and policy simulation through FastAPI.
10. Upgrade the React/Vite dashboard with portfolio KPIs, filters, customer detail modal, budget optimizer and policy simulator.
11. Add API and business-rule tests.
12. Validate the backend with a live Uvicorn smoke test.
13. Keep Docker packaging available for deployment.

## Deliberately not added

- No database: the current artifact is a stateless portfolio analytics application.
- No authentication: not needed for the local portfolio scope.
- No deep-learning model: it would add complexity without a demonstrated business/evaluation need here.
- No unrelated technologies purely for resume keywords.

## Validation target

The final release should pass:

```bash
pytest -q
python train.py
cd frontend && npm install && npm run build
```

The backend should expose OpenAPI documentation at `/docs` and all documented routes should return valid responses with the trained artifacts present.
