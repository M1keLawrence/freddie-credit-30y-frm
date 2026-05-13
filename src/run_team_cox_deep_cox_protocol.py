from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deep_cox_assignment import (
    DeepCox,
    concordance_index,
    estimate_breslow_baseline_hazard,
    extract_survival_targets,
    fit_classical_cox,
    fit_survival_preprocessor,
    predict_classical_cox_log_risk,
    predict_deep_cox_log_risk_batched,
    predict_event_probability_at_horizon,
    prepare_classical_cox_frame,
    prepare_classical_cox_splits,
    read_survival_table,
    set_seed,
    split_train_validation_dataframe,
    to_tensor_frame,
    to_tensor_vector,
    train_deep_cox,
    transform_survival_features,
    validate_survival_columns,
)


DATA_DIR = Path("IR")
TRAIN_PATH = DATA_DIR / "team_train.parquet"
EVAL_SPECS = [
    {
        "name": "test",
        "path": DATA_DIR / "team_test.parquet",
        "horizon": 60,
        "label_col": "y_prepay_60m",
        "cox_output": Path("cox_test_predictions.csv"),
        "deep_output": Path("deep_cox_test_predictions.csv"),
        "cox_prob_col": "coxprob_prepay60mo",
        "deep_prob_col": "deep_coxprob_prepay60mo",
    },
    {
        "name": "backtest_a",
        "path": DATA_DIR / "team_backtest_a.parquet",
        "horizon": 24,
        "label_col": "y_prepay_24m",
        "cox_output": Path("cox_backtest_a_predictions.csv"),
        "deep_output": Path("deep_cox_backtest_a_predictions.csv"),
        "cox_prob_col": "coxprob_prepay24mo",
        "deep_prob_col": "deep_coxprob_prepay24mo",
    },
    {
        "name": "backtest_b",
        "path": DATA_DIR / "team_backtest_b.parquet",
        "horizon": 12,
        "label_col": "y_prepay_12m",
        "cox_output": Path("cox_backtest_b_predictions.csv"),
        "deep_output": Path("deep_cox_backtest_b_predictions.csv"),
        "cox_prob_col": "coxprob_prepay12mo",
        "deep_prob_col": "deep_coxprob_prepay12mo",
    },
]

ID_COL = "loan_seq_num"
TIME_COL = "event_time_months"
EVENT_COL = "prepay_observed"
CATEGORICAL_COLS = [
    "state",
    "occupancy",
    "loan_purpose",
    "channel",
    "first_time_homebuyer",
    "prop_type",
]

VAL_SIZE = 0.15
SEED = 42

HIDDEN_DIMS = (128, 64, 32)
DROPOUT = 0.05
ACTIVATION = "gelu"
USE_BATCH_NORM = True
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 150
PATIENCE = 20
CINDEX_EVERY = 5
SELECTION_METRIC = "cindex"
SELECTION_MIN_DELTA = 1e-4
COMPUTE_TRAIN_CINDEX = False
TIE_METHOD = "efron"
COX_PENALIZER = 1e-3
COX_L1_RATIO = 0.0
DEEP_PRED_BATCH_SIZE = 200_000


def infer_feature_columns(train_df: pd.DataFrame) -> list[str]:
    excluded = {
        ID_COL,
        TIME_COL,
        EVENT_COL,
        "y_prepay_60m",
        "y_prepay_24m",
        "y_prepay_12m",
    }
    feature_cols = [col for col in train_df.columns if col not in excluded]
    if not feature_cols:
        raise ValueError("No feature columns were inferred from team_train.parquet.")
    return feature_cols


def build_submission_frame(
    loan_ids: pd.Series,
    probabilities: np.ndarray,
    probability_column: str,
) -> pd.DataFrame:
    submission = pd.DataFrame(
        {
            ID_COL: loan_ids.to_numpy(copy=False),
            probability_column: probabilities.astype(np.float64, copy=False),
        }
    )
    return submission


def log_dataset_overview(name: str, df: pd.DataFrame, label_col: str | None = None) -> None:
    print(f"{name}: shape={df.shape}")
    if label_col is not None and label_col in df.columns:
        print(f"  {label_col} mean={float(df[label_col].mean()):.6f}")
    print(f"  {EVENT_COL} mean={float(df[EVENT_COL].mean()):.6f}")
    print(
        f"  {TIME_COL} range=({int(df[TIME_COL].min())}, {int(df[TIME_COL].max())})"
    )


def main() -> None:
    set_seed(SEED)
    start_time = time.perf_counter()

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing training file: {TRAIN_PATH.resolve()}")

    print(f"Loading training data from {TRAIN_PATH.resolve()}")
    train_full_df = read_survival_table(TRAIN_PATH)
    feature_cols = infer_feature_columns(train_full_df)
    validate_survival_columns(train_full_df, TIME_COL, EVENT_COL, feature_cols)
    log_dataset_overview("team_train_full", train_full_df, label_col="y_prepay_60m")
    print(f"Inferred feature count: {len(feature_cols)}")
    print(f"Categorical columns: {CATEGORICAL_COLS}")

    train_df, val_df = split_train_validation_dataframe(
        df=train_full_df,
        event_col=EVENT_COL,
        val_size=VAL_SIZE,
        random_state=SEED,
    )
    log_dataset_overview("team_train_split", train_df, label_col="y_prepay_60m")
    log_dataset_overview("team_val_split", val_df, label_col="y_prepay_60m")

    preprocessor = fit_survival_preprocessor(
        train_df=train_df,
        feature_cols=feature_cols,
        categorical_cols=CATEGORICAL_COLS,
    )
    x_train = transform_survival_features(train_df, preprocessor)
    x_val = transform_survival_features(val_df, preprocessor)
    t_train, e_train = extract_survival_targets(train_df, TIME_COL, EVENT_COL)
    t_val, e_val = extract_survival_targets(val_df, TIME_COL, EVENT_COL)

    classical_prepared = prepare_classical_cox_splits(
        x_train=x_train,
        x_val=x_val,
        x_test=x_val,
        categorical_columns=CATEGORICAL_COLS,
    )

    print(
        "Prepared feature matrices: "
        f"deep={x_train.shape[1]} columns, classical={classical_prepared.x_train.shape[1]} columns"
    )

    classical_start = time.perf_counter()
    classical_cox = fit_classical_cox(
        x_train=classical_prepared.x_train,
        time_train=t_train,
        event_train=e_train,
        penalizer=COX_PENALIZER,
        l1_ratio=COX_L1_RATIO,
    )
    classical_seconds = time.perf_counter() - classical_start
    print(f"Classical Cox fit completed in {classical_seconds:.2f}s")

    x_train_tensor = to_tensor_frame(x_train)
    x_val_tensor = to_tensor_frame(x_val)
    t_train_tensor = to_tensor_vector(t_train)
    t_val_tensor = to_tensor_vector(t_val)
    e_train_tensor = to_tensor_vector(e_train)
    e_val_tensor = to_tensor_vector(e_val)

    model = DeepCox(
        input_dim=x_train_tensor.shape[1],
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
        activation=ACTIVATION,
        use_batch_norm=USE_BATCH_NORM,
    )
    deep_start = time.perf_counter()
    model, history = train_deep_cox(
        model=model,
        x_train=x_train_tensor,
        t_train=t_train_tensor,
        e_train=e_train_tensor,
        x_val=x_val_tensor,
        t_val=t_val_tensor,
        e_val=e_val_tensor,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        cindex_every=CINDEX_EVERY,
        selection_metric=SELECTION_METRIC,
        selection_min_delta=SELECTION_MIN_DELTA,
        compute_train_cindex=COMPUTE_TRAIN_CINDEX,
        tie_method=TIE_METHOD,
        log_progress=True,
    )
    deep_seconds = time.perf_counter() - deep_start
    print(f"Deep Cox training completed in {deep_seconds:.2f}s")

    with torch.no_grad():
        deep_train_log_risk = model(x_train_tensor).detach().cpu().numpy().reshape(-1)
        deep_val_log_risk = model(x_val_tensor).detach().cpu().numpy().reshape(-1)
    classical_val_log_risk = predict_classical_cox_log_risk(
        classical_cox,
        classical_prepared.x_val,
    )
    print(
        "Validation C-index: "
        f"classical={concordance_index(t_val, e_val, classical_val_log_risk):.6f}, "
        f"deep={concordance_index(t_val, e_val, deep_val_log_risk):.6f}"
    )

    baseline_hazard = estimate_breslow_baseline_hazard(
        time=t_train,
        event=e_train,
        log_risk=deep_train_log_risk,
    )

    del x_train_tensor
    del x_val_tensor
    del t_train_tensor
    del t_val_tensor
    del e_train_tensor
    del e_val_tensor
    gc.collect()

    for spec in EVAL_SPECS:
        eval_start = time.perf_counter()
        print(f"\nScoring {spec['name']} from {spec['path'].resolve()}")
        eval_df = read_survival_table(spec["path"])
        validate_survival_columns(eval_df, TIME_COL, EVENT_COL, feature_cols)
        log_dataset_overview(spec["name"], eval_df, label_col=spec["label_col"])

        x_eval = transform_survival_features(eval_df, preprocessor)
        x_eval_classical = prepare_classical_cox_frame(
            x_eval,
            classical_prepared.dropped_reference_columns,
        )

        classical_log_risk = predict_classical_cox_log_risk(classical_cox, x_eval_classical)
        classical_prob = predict_event_probability_at_horizon(
            baseline_hazard=classical_cox.baseline_cumulative_hazard_,
            log_risk=classical_log_risk,
            horizon=spec["horizon"],
        )
        classical_submission = build_submission_frame(
            loan_ids=eval_df[ID_COL],
            probabilities=classical_prob,
            probability_column=spec["cox_prob_col"],
        )
        classical_submission.to_csv(spec["cox_output"], index=False)
        print(f"  Saved {spec['cox_output'].resolve()}")

        deep_log_risk = predict_deep_cox_log_risk_batched(
            model=model,
            x=x_eval,
            batch_size=DEEP_PRED_BATCH_SIZE,
        )
        deep_prob = predict_event_probability_at_horizon(
            baseline_hazard=baseline_hazard,
            log_risk=deep_log_risk,
            horizon=spec["horizon"],
        )
        deep_submission = build_submission_frame(
            loan_ids=eval_df[ID_COL],
            probabilities=deep_prob,
            probability_column=spec["deep_prob_col"],
        )
        deep_submission.to_csv(spec["deep_output"], index=False)
        print(f"  Saved {spec['deep_output'].resolve()}")

        elapsed = time.perf_counter() - eval_start
        print(
            f"  Probability means: classical={classical_prob.mean():.6f}, "
            f"deep={deep_prob.mean():.6f}"
        )
        print(f"  Finished {spec['name']} in {elapsed:.2f}s")

        del eval_df
        del x_eval
        del x_eval_classical
        del classical_log_risk
        del deep_log_risk
        del classical_prob
        del deep_prob
        del classical_submission
        del deep_submission
        gc.collect()

    total_seconds = time.perf_counter() - start_time
    print("\nDone.")
    print(f"Total elapsed time: {total_seconds:.2f}s")
    print(f"Deep Cox history rows: {len(history)}")
    if not history.empty:
        print(
            f"Best checkpoint epoch: {int(history['best_epoch'].iloc[-1])}, "
            f"best validation C-index: {float(history['best_val_cindex'].iloc[-1]):.6f}"
        )


if __name__ == "__main__":
    main()
