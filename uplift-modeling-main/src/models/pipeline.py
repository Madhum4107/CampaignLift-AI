from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import load_hillstrom, add_binary_treatment
from src.preprocessing.features import build_preprocessor, feature_frame
from src.models.uplift import SLearner, TLearner, XLearner
from src.evaluation.metrics import evaluate_uplift


class UpliftPipeline:
    """Train and compare uplift learners with a held-out test split.

    Preprocessing is fit only on the training split for model selection, then
    refit on train+validation before the final held-out test evaluation.
    """

    def __init__(self, data_path="data/hillstrom.csv", artifact_dir="models"):
        self.data_path = Path(data_path)
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def fit(self):
        df = add_binary_treatment(load_hillstrom(self.data_path))
        y = df.conversion.to_numpy()
        treatment = df.treatment.to_numpy()
        raw = feature_frame(df)
        indices = np.arange(len(df))
        strata = treatment.astype(str) + y.astype(str)

        train_idx, remaining = train_test_split(
            indices, test_size=0.30, random_state=42, stratify=strata
        )
        val_idx, test_idx = train_test_split(
            remaining, test_size=0.50, random_state=42, stratify=strata[remaining]
        )

        # Fit transforms on training data only for validation/model selection.
        pre_validation = build_preprocessor()
        X_train = pre_validation.fit_transform(raw.iloc[train_idx])
        X_val = pre_validation.transform(raw.iloc[val_idx])
        models = [SLearner(), TLearner(), XLearner()]
        results, fitted = {}, {}
        for learner in models:
            learner.fit(X_train, treatment[train_idx], y[train_idx])
            validation = evaluate_uplift(
                y[val_idx], treatment[val_idx], learner.predict_uplift(X_val)
            )
            results[learner.name] = {
                key: value for key, value in validation.items() if key != "curve"
            }
            fitted[learner.name] = learner

        best = max(results, key=lambda name: results[name]["qini_coefficient"])

        # Refit preprocessing and selected learner using train + validation;
        # keep the test set untouched until final evaluation.
        development_idx = np.concatenate([train_idx, val_idx])
        pre_final = build_preprocessor()
        X_development = pre_final.fit_transform(raw.iloc[development_idx])
        X_test = pre_final.transform(raw.iloc[test_idx])
        selected = type(fitted[best])()
        selected.fit(X_development, treatment[development_idx], y[development_idx])
        test_eval = evaluate_uplift(
            y[test_idx], treatment[test_idx], selected.predict_uplift(X_test)
        )
        results[best].update({
            "test_auuc": test_eval["auuc"],
            "test_qini_coefficient": test_eval["qini_coefficient"],
        })

        # Score rows for the demo dashboard; the test rows are inference-only.
        X_all = pre_final.transform(raw)
        uplift = selected.predict_uplift(X_all)
        if best == "T-Learner":
            control_probability = selected.control.predict_proba(X_all)[:, 1]
        elif best == "X-Learner":
            control_probability = selected.m0.predict_proba(X_all)[:, 1]
        else:
            control_probability = selected.model.predict_proba(
                np.column_stack([X_all, np.zeros(len(X_all))])
            )[:, 1]

        customers = df[["recency", "history", "channel", "segment"]].copy()
        customers.insert(0, "customer_id", np.arange(len(df)))
        customers["uplift"] = uplift
        customers["p_control"] = control_probability
        from src.optimization.business import assign_segments
        customers = assign_segments(customers)
        customers.to_csv(self.artifact_dir / "customer_scores.csv", index=False)
        joblib.dump(
            {"preprocessor": pre_final, "model": selected, "best_model": best},
            self.artifact_dir / "uplift_model.joblib",
        )

        curves = {}
        for name, learner in fitted.items():
            curve = evaluate_uplift(
                y[val_idx], treatment[val_idx], learner.predict_uplift(X_val)
            )["curve"]
            curves[name] = curve.iloc[::max(1, len(curve) // 100)].to_dict("records")

        metrics_payload = {
            "models": results,
            "best_model": best,
            "curves": curves,
            "training": {
                "rows": len(df),
                "train_rows": len(train_idx),
                "validation_rows": len(val_idx),
                "test_rows": len(test_idx),
                "treatment_rate": float(treatment.mean()),
                "seed": 42,
                "preprocessing_fit": "train for validation; train+validation for final model",
            },
        }
        (self.artifact_dir / "metrics.json").write_text(
            json.dumps(metrics_payload, indent=2), encoding="utf-8"
        )
        return results, best


def load_artifacts(artifact_dir="models"):
    directory = Path(artifact_dir)
    required = ["uplift_model.joblib", "customer_scores.csv", "metrics.json"]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model artifacts: {missing}")
    bundle = joblib.load(directory / "uplift_model.joblib")
    customers = pd.read_csv(directory / "customer_scores.csv")
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    return bundle, customers, metrics
