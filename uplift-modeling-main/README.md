# CampaignLift AI

CampaignLift AI is an end-to-end uplift-modeling portfolio application for deciding **who should receive a marketing intervention based on estimated incremental response**, rather than purchase probability alone.

## What the final version demonstrates

- Randomized treatment/control experiment analysis for the Hillstrom email dataset
- Leakage-aware train/validation/test separation
- S-, T-, and X-Learner uplift modeling
- Qini curve, AUUC, Qini coefficient and uplift-at-k evaluation
- Holdout evaluation for the selected model
- Individual customer uplift scoring and four-quadrant-style business segmentation
- Customer-level model-sensitivity explanation (explicitly not a causal explanation)
- Budget-constrained targeting
- Campaign guardrails (minimum uplift, maximum contacts and segment exclusions)
- Campaign readiness pre-flight checks
- Data/model monitoring health endpoint with expected incremental value
- Campaign policy simulation (Random vs Top 5/10/20/30%)
- Search/filterable target-customer dashboard
- Segment intelligence cards with population share, average uplift and recommended action
- Budget scenario sweep at 25/50/75/100% of the entered budget
- CSV export of filtered target candidates for campaign handoff
- FastAPI service with OpenAPI docs
- React/Vite business dashboard
- Pytest coverage for data, metrics, business rules and API routes
- Docker packaging

## Important modeling choices

The target outcome is binary conversion. The treatment is **any email vs No E-Mail** for the production uplift model; the original Mens E-Mail and Womens E-Mail arms remain available for experiment analysis.

The outcome models intentionally use probability-calibrated classifiers without class rebalancing weights. This matters for uplift: class weighting can distort the absolute probability scale when conversion is rare, which can produce implausibly large treatment-effect estimates. Individual uplift is bounded to the mathematically valid `[-1, 1]` interval.

The current artifact was trained with a fixed random seed (`42`) and a train/validation/test split. The model is selected using validation Qini coefficient, then refit on train+validation and evaluated once on the isolated test split.

## Dashboard features

The React dashboard includes:

1. Experiment summary — Mens/Womens email conversion and lift.
2. Portfolio KPIs — customer count, positive-uplift rate, targetable population and mean uplift.
3. Model comparison — AUUC, Qini and uplift-at-20 with holdout Qini for the selected model.
4. Target candidates — sortable API-backed ranking with segment filter and minimum-uplift filter.
5. Customer profile — uplift, control probability, segment, recommendation and local sensitivity drivers.
6. Budget optimizer — budget/cost/revenue assumptions converted into a targeting policy.
7. Policy simulator — broad random contact vs uplift-ranked policies.
8. Segment intelligence — population composition, average uplift and control probability by segment.
9. Budget scenarios — stress-test campaign economics at four spend levels.
10. CSV export — download the current target list with model and customer context.

## Dataset

Kevin Hillstrom MineThatData Email Analytics Challenge. The supplied CSV contains 64,000 randomized customers across No E-Mail, Mens E-Mail and Womens E-Mail groups.

## Architecture

```text
React/Vite dashboard
        |
        v
FastAPI REST API
        |
        +--> experiment statistics
        +--> customer ranking / explanations
        +--> budget optimizer
        +--> policy simulator
        |
        v
Uplift model artifacts
        |
        v
Hillstrom CSV
```

## Run locally

### Backend

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
python train.py
python -m uvicorn backend.main:app --reload --port 8000
```

Open:

- `http://localhost:8000/` — API root
- `http://localhost:8000/health` — health check
- `http://localhost:8000/docs` — interactive OpenAPI documentation

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL printed in the terminal. The dashboard defaults to `http://localhost:8000` for the API. To override it, create `frontend/.env.local`:

```text
VITE_API_URL=http://localhost:8000
```

### Production frontend check

```bash
npm run build
npm run preview
```

Use `vite build` as the production build command and `vite preview` to inspect the built output locally.

### Tests

```bash
pytest -q
```

The API tests use FastAPI TestClient against the application routes.

### Docker

```bash
docker compose up --build
```

## API endpoints

- `GET /`
- `GET /health`
- `GET /summary`
- `GET /experiment/stats`
- `GET /model/metrics`
- `GET /customers`
- `GET /customers/{customer_id}`
- `GET /customers/export`
- `GET /segments`
- `POST /optimize/budget`
- `POST /optimize/budget/sweep`
- `POST /simulate/policy`

## Interpretation and limitations

Uplift estimates are model outputs, not guaranteed incremental sales or profit. Qini/AUUC measure how well the model ranks treatment effect under the evaluation protocol; they are not ordinary classification accuracy.

The production-style use case should be validated with fresh randomized campaign experiments. Individual treatment-effect estimates can be noisy, and the current binary treatment combines Mens and Womens Email. A future extension could model the two email treatments separately.

Customer explanations use local sensitivity analysis: a feature is replaced with a population reference value and the change in predicted uplift is measured. This describes model sensitivity; it does not prove a feature caused the treatment effect.

## Portfolio positioning

This project is intentionally a **focused end-to-end Data Science / ML portfolio application**, not a collection of unrelated technologies. It demonstrates experimentation, causal/uplift modeling, model evaluation, business decision logic, API engineering, frontend visualization, testing and reproducibility in one coherent use case.

## License

This repository is a student/portfolio implementation based on the supplied reference repository. Review the original repository license and dataset terms before redistributing the source or dataset.
