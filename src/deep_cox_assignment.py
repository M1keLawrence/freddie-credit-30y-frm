from __future__ import annotations

import copy
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


_WARNED_ABOUT_APPROX_CINDEX = False


def _missing_lifelines_message() -> str:
    project_root = Path(__file__).resolve().parent
    venv_python = project_root / ".venv" / "bin" / "python"
    requirements_file = project_root / "requirements-deep-cox.txt"

    lines = [
        "lifelines is required to fit the classical Cox benchmark.",
        f"Current Python executable: {sys.executable}",
    ]

    if venv_python.exists():
        lines.append(
            f"Project virtualenv detected at: {venv_python}"
        )
        lines.append(
            "If you are in Jupyter or VS Code, switch the notebook kernel to "
            "'Python (.venv Deep Cox)'."
        )

    if requirements_file.exists():
        if venv_python.exists():
            lines.append(
                "To install into the project virtualenv, run: "
                f"{venv_python} -m pip install -r {requirements_file}"
            )
        else:
            lines.append(
                "Install project dependencies with: "
                f"python -m pip install -r {requirements_file}"
            )

    return " ".join(lines)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_survival_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at {path.resolve()}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(
        "Unsupported file format. Use a .csv, .xlsx, .xls, or .parquet file."
    )


def validate_survival_columns(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    feature_cols: Sequence[str] | None = None,
) -> list[str]:
    required = [time_col, event_col]
    if feature_cols is not None:
        required.extend(feature_cols)

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    event_values = set(pd.Series(df[event_col]).dropna().unique())
    if not event_values.issubset({0, 1, False, True}):
        raise ValueError(
            f"`{event_col}` must be binary with 1=event and 0=censored. Found {event_values}."
        )

    inferred_feature_cols = (
        list(feature_cols)
        if feature_cols is not None
        else [col for col in df.columns if col not in {time_col, event_col}]
    )
    if not inferred_feature_cols:
        raise ValueError("No feature columns were selected.")

    return inferred_feature_cols


def _safe_stratify(labels: pd.Series) -> pd.Series | None:
    counts = labels.value_counts(dropna=False)
    if counts.shape[0] < 2 or (counts < 2).any():
        return None
    return labels


def _chronological_period_boundaries(
    ordered_periods: pd.Series,
    val_size: float,
    test_size: float,
) -> tuple[object, object]:
    unique_periods = ordered_periods.drop_duplicates().to_list()
    n_periods = len(unique_periods)
    if n_periods < 3:
        raise ValueError(
            "A chronological split requires at least three distinct time periods."
        )

    period_counts = ordered_periods.groupby(ordered_periods, sort=False).size()
    cumulative_counts = period_counts.cumsum().to_numpy()
    total_rows = int(cumulative_counts[-1])

    train_target = total_rows * (1.0 - val_size - test_size)
    val_target = total_rows * (1.0 - test_size)

    train_last_idx = int(np.searchsorted(cumulative_counts, train_target, side="left"))
    val_last_idx = int(np.searchsorted(cumulative_counts, val_target, side="left"))

    train_last_idx = min(max(train_last_idx, 0), n_periods - 3)
    val_last_idx = min(max(val_last_idx, train_last_idx + 1), n_periods - 2)

    return unique_periods[train_last_idx + 1], unique_periods[val_last_idx + 1]


def split_survival_dataframe(
    df: pd.DataFrame,
    event_col: str,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    sort_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("Use val_size > 0, test_size > 0, and val_size + test_size < 1.")

    if sort_col is not None:
        if sort_col not in df.columns:
            raise KeyError(f"Missing chronological split column: {sort_col}")
        if df[sort_col].isna().any():
            raise ValueError(
                f"Chronological split column `{sort_col}` contains missing values."
            )

        ordered_df = df.sort_values(sort_col, kind="stable").reset_index(drop=True)
        train_cutoff, val_cutoff = _chronological_period_boundaries(
            ordered_df[sort_col],
            val_size=val_size,
            test_size=test_size,
        )

        train_df = ordered_df[ordered_df[sort_col] < train_cutoff]
        val_df = ordered_df[
            (ordered_df[sort_col] >= train_cutoff) & (ordered_df[sort_col] < val_cutoff)
        ]
        test_df = ordered_df[ordered_df[sort_col] >= val_cutoff]

        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError(
                "Chronological split produced an empty train, validation, or test set."
            )

        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    stratify = _safe_stratify(df[event_col])
    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        random_state=random_state,
        stratify=stratify,
    )

    relative_test_size = test_size / (val_size + test_size)
    temp_stratify = _safe_stratify(temp_df[event_col])
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=temp_stratify,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


@dataclass
class PreparedSurvivalData:
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    t_train: np.ndarray
    t_val: np.ndarray
    t_test: np.ndarray
    e_train: np.ndarray
    e_val: np.ndarray
    e_test: np.ndarray
    feature_names: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]


@dataclass
class PreparedClassicalCoxData:
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    dropped_reference_columns: list[str]


def prepare_survival_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    time_col: str,
    event_col: str,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str] | None = None,
) -> PreparedSurvivalData:
    feature_cols = list(feature_cols)
    categorical_cols = (
        list(categorical_cols)
        if categorical_cols is not None
        else [
            col
            for col in feature_cols
            if pd.api.types.is_object_dtype(train_df[col])
            or isinstance(train_df[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(train_df[col])
        ]
    )
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    if numeric_cols:
        train_num_raw = train_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        medians = train_num_raw.median()
        scaler = StandardScaler().fit(train_num_raw.fillna(medians))

        def _transform_numeric(df: pd.DataFrame) -> pd.DataFrame:
            numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(medians)
            values = scaler.transform(numeric)
            return pd.DataFrame(values, columns=numeric_cols, index=df.index)

        train_num = _transform_numeric(train_df)
        val_num = _transform_numeric(val_df)
        test_num = _transform_numeric(test_df)
    else:
        train_num = pd.DataFrame(index=train_df.index)
        val_num = pd.DataFrame(index=val_df.index)
        test_num = pd.DataFrame(index=test_df.index)

    def _prepare_categorical(df: pd.DataFrame, train_columns: Sequence[str] | None = None):
        if not categorical_cols:
            return pd.DataFrame(index=df.index)

        categorical = (
            df[categorical_cols]
            .fillna("__missing__")
            .astype("string")
        )
        encoded = pd.get_dummies(categorical, columns=categorical_cols, dtype=float)
        if train_columns is not None:
            encoded = encoded.reindex(columns=train_columns, fill_value=0.0)
        return encoded

    train_cat = _prepare_categorical(train_df)
    val_cat = _prepare_categorical(val_df, train_columns=train_cat.columns)
    test_cat = _prepare_categorical(test_df, train_columns=train_cat.columns)

    x_train = pd.concat([train_num, train_cat], axis=1).astype(np.float32)
    x_val = pd.concat([val_num, val_cat], axis=1).astype(np.float32)
    x_test = pd.concat([test_num, test_cat], axis=1).astype(np.float32)

    feature_names = list(x_train.columns)
    x_val = x_val.reindex(columns=feature_names, fill_value=0.0)
    x_test = x_test.reindex(columns=feature_names, fill_value=0.0)

    return PreparedSurvivalData(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        t_train=train_df[time_col].to_numpy(dtype=np.float32),
        t_val=val_df[time_col].to_numpy(dtype=np.float32),
        t_test=test_df[time_col].to_numpy(dtype=np.float32),
        e_train=train_df[event_col].to_numpy(dtype=np.float32),
        e_val=val_df[event_col].to_numpy(dtype=np.float32),
        e_test=test_df[event_col].to_numpy(dtype=np.float32),
        feature_names=feature_names,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
    )


def prepare_classical_cox_splits(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    categorical_columns: Sequence[str] | None = None,
) -> PreparedClassicalCoxData:
    dropped_reference_columns: list[str] = []

    if categorical_columns is not None:
        for categorical_column in categorical_columns:
            matching_columns = [
                column
                for column in x_train.columns
                if column.startswith(f"{categorical_column}_")
            ]
            if len(matching_columns) > 1:
                dropped_reference_columns.append(matching_columns[0])

    return PreparedClassicalCoxData(
        x_train=x_train.drop(columns=dropped_reference_columns, errors="ignore").copy(),
        x_val=x_val.drop(columns=dropped_reference_columns, errors="ignore").copy(),
        x_test=x_test.drop(columns=dropped_reference_columns, errors="ignore").copy(),
        dropped_reference_columns=dropped_reference_columns,
    )


class DeepCox(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64, 32),
        dropout: float = 0.05,
        activation: str = "gelu",
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim

        activation_layer: type[nn.Module]
        activation_name = activation.lower()
        if activation_name == "gelu":
            activation_layer = nn.GELU
        elif activation_name == "relu":
            activation_layer = nn.ReLU
        else:
            raise ValueError("activation must be either 'gelu' or 'relu'.")

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation_layer())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def cox_ph_loss(
    log_risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    tie_method: str = "efron",
) -> torch.Tensor:
    if log_risk.ndim != 1 or time.ndim != 1 or event.ndim != 1:
        raise ValueError("log_risk, time, and event must all be 1D tensors.")
    if not (log_risk.shape[0] == time.shape[0] == event.shape[0]):
        raise ValueError("log_risk, time, and event must have the same length.")
    if tie_method not in {"breslow", "efron"}:
        raise ValueError("tie_method must be either 'breslow' or 'efron'.")

    event = event.float()
    if event.sum() == 0:
        raise ValueError("At least one observed event is required to compute Cox loss.")

    order = torch.argsort(time, descending=True)
    time_sorted = time[order]
    event_sorted = event[order]
    log_risk_sorted = log_risk[order]

    hazard_ratio = torch.exp(log_risk_sorted)
    cumulative_risk = torch.cumsum(hazard_ratio, dim=0)

    counts = torch.unique_consecutive(time_sorted, return_counts=True)[1]
    end_indices = torch.cumsum(counts, dim=0) - 1

    group_losses: list[torch.Tensor] = []
    start = 0
    for count, end_index in zip(counts.tolist(), end_indices):
        stop = start + count
        group_event = event_sorted[start:stop]
        event_count = int(group_event.sum().item())
        if event_count > 0:
            group_log_risk = log_risk_sorted[start:stop]
            group_hazard_ratio = hazard_ratio[start:stop]
            event_mask = group_event.bool()
            observed_log_risk = group_log_risk[event_mask].sum()
            risk_set_sum = cumulative_risk[end_index]

            if tie_method == "breslow" or event_count == 1:
                group_losses.append(
                    observed_log_risk - event_count * torch.log(risk_set_sum)
                )
            else:
                tied_hazard_sum = group_hazard_ratio[event_mask].sum()
                efron_steps = torch.arange(
                    event_count,
                    device=log_risk.device,
                    dtype=log_risk.dtype,
                )
                adjusted_risk = risk_set_sum - (efron_steps / event_count) * tied_hazard_sum
                adjusted_risk = adjusted_risk.clamp_min(1e-12)
                group_losses.append(observed_log_risk - torch.log(adjusted_risk).sum())
        start = stop

    partial_log_likelihood = torch.stack(group_losses).sum()
    return -partial_log_likelihood / event.sum()


def _pairwise_concordance_index(
    time: Sequence[float],
    event: Sequence[float],
    risk: Sequence[float],
) -> float:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    risk = np.asarray(risk, dtype=float)

    permissible = 0.0
    concordant = 0.0
    tied = 0.0

    n_samples = len(time)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if time[i] == time[j]:
                continue

            if event[i] == 1 and time[i] < time[j]:
                permissible += 1
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    tied += 1
            elif event[j] == 1 and time[j] < time[i]:
                permissible += 1
                if risk[j] > risk[i]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    tied += 1

    if permissible == 0:
        return float("nan")
    return float((concordant + 0.5 * tied) / permissible)


def concordance_index(time: Sequence[float], event: Sequence[float], risk: Sequence[float]) -> float:
    global _WARNED_ABOUT_APPROX_CINDEX

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    risk = np.asarray(risk, dtype=float)

    if len(time) == 0:
        return float("nan")

    try:
        from lifelines.utils import concordance_index as lifelines_concordance_index

        # lifelines expects larger predicted scores to imply longer survival,
        # while our model output is a log-risk score where larger means shorter survival.
        return float(lifelines_concordance_index(time, -risk, event))
    except ModuleNotFoundError:
        max_samples = 2000
        if len(time) > max_samples:
            rng = np.random.default_rng(12345)
            sample_idx = rng.choice(len(time), size=max_samples, replace=False)
            time = time[sample_idx]
            event = event[sample_idx]
            risk = risk[sample_idx]

        if not _WARNED_ABOUT_APPROX_CINDEX:
            print(
                "lifelines is not installed in this kernel. "
                "Using a smaller approximate C-index sample for progress logging."
            )
            _WARNED_ABOUT_APPROX_CINDEX = True

        return _pairwise_concordance_index(time, event, risk)


def to_tensor_frame(frame: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(frame.to_numpy(dtype=np.float32), dtype=torch.float32)


def to_tensor_vector(values: Sequence[float]) -> torch.Tensor:
    return torch.tensor(np.asarray(values, dtype=np.float32), dtype=torch.float32)


def fit_classical_cox(
    x_train: pd.DataFrame,
    time_train: Sequence[float],
    event_train: Sequence[float],
    penalizer: float = 1e-3,
    l1_ratio: float = 0.0,
    show_progress: bool = False,
):
    try:
        from lifelines import CoxPHFitter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_missing_lifelines_message()) from exc

    train_frame = x_train.astype(np.float32).copy()
    train_frame["_time"] = np.asarray(time_train, dtype=float)
    train_frame["_event"] = np.asarray(event_train, dtype=int)

    model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    model.fit(
        train_frame,
        duration_col="_time",
        event_col="_event",
        show_progress=show_progress,
    )
    return model


def predict_classical_cox_log_risk(model, x: pd.DataFrame) -> np.ndarray:
    log_risk = model.predict_log_partial_hazard(x.astype(np.float32))
    if isinstance(log_risk, pd.Series):
        return log_risk.to_numpy(dtype=float)
    return np.asarray(log_risk, dtype=float).reshape(-1)


def train_deep_cox(
    model: DeepCox,
    x_train: torch.Tensor,
    t_train: torch.Tensor,
    e_train: torch.Tensor,
    x_val: torch.Tensor,
    t_val: torch.Tensor,
    e_val: torch.Tensor,
    epochs: int = 500,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 50,
    verbose_every: int = 25,
    cindex_every: int = 5,
    compute_train_cindex: bool = False,
    tie_method: str = "efron",
    selection_metric: str = "cindex",
    selection_min_delta: float | None = None,
    log_progress: bool = True,
) -> tuple[DeepCox, pd.DataFrame]:
    if cindex_every <= 0:
        raise ValueError("cindex_every must be a positive integer.")
    if selection_metric not in {"loss", "cindex"}:
        raise ValueError("selection_metric must be either 'loss' or 'cindex'.")

    if selection_min_delta is None:
        selection_min_delta = 1e-4 if selection_metric == "cindex" else 1e-6

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_loss = np.inf
    best_val_cindex = float("-inf")
    best_selection_score = np.inf if selection_metric == "loss" else float("-inf")
    wait = 0
    history: list[dict[str, float]] = []
    training_start = time.perf_counter()
    train_event_count = int(e_train.sum().item())
    val_event_count = int(e_val.sum().item())
    t_train_np = t_train.detach().cpu().numpy()
    e_train_np = e_train.detach().cpu().numpy()
    t_val_np = t_val.detach().cpu().numpy()
    e_val_np = e_val.detach().cpu().numpy()
    last_train_cindex = float("nan")
    last_val_cindex = float("nan")

    if log_progress:
        print(
            "Starting Deep Cox training "
            f"(epochs={epochs}, lr={learning_rate:.1e}, weight_decay={weight_decay:.1e}, "
            f"patience={patience}, cindex_every={cindex_every}, "
            f"compute_train_cindex={compute_train_cindex}, "
            f"tie_method={tie_method}, "
            f"selection_metric={selection_metric}, selection_min_delta={selection_min_delta:.1e}, "
            f"train_rows={x_train.shape[0]:,}, val_rows={x_val.shape[0]:,}, "
            f"features={x_train.shape[1]}, train_events={train_event_count:,}, val_events={val_event_count:,})."
        )

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        train_log_risk = model(x_train)
        train_loss = cox_ph_loss(train_log_risk, t_train, e_train, tie_method=tie_method)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_log_risk = model(x_train)
            val_log_risk = model(x_val)
            val_loss = cox_ph_loss(val_log_risk, t_val, e_val, tie_method=tie_method)

        computed_cindex = (
            epoch == 1
            or epoch == epochs
            or epoch % cindex_every == 0
        )

        if computed_cindex:
            val_cindex = concordance_index(
                t_val_np,
                e_val_np,
                val_log_risk.detach().cpu().numpy(),
            )
            last_val_cindex = float(val_cindex)

            if compute_train_cindex:
                train_cindex = concordance_index(
                    t_train_np,
                    e_train_np,
                    train_log_risk.detach().cpu().numpy(),
                )
                last_train_cindex = float(train_cindex)
        val_loss_value = float(val_loss.item())

        if selection_metric == "loss":
            improved = val_loss_value < (best_selection_score - selection_min_delta)
            if improved:
                best_selection_score = val_loss_value
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                wait = 0
            else:
                wait = epoch - best_epoch
        else:
            improved = False
            if computed_cindex:
                improved = last_val_cindex > (best_selection_score + selection_min_delta)
                if improved:
                    best_selection_score = last_val_cindex
                    best_val_cindex = last_val_cindex
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                wait = epoch - best_epoch
            else:
                wait = epoch - best_epoch

        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value

        if selection_metric == "loss" and computed_cindex and last_val_cindex > best_val_cindex:
            best_val_cindex = last_val_cindex

        epoch_seconds = time.perf_counter() - epoch_start
        elapsed_seconds = time.perf_counter() - training_start

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss.item()),
                "val_loss": val_loss_value,
                "train_cindex": float(last_train_cindex),
                "val_cindex": float(last_val_cindex),
                "best_val_loss": float(best_val_loss),
                "best_val_cindex": float(best_val_cindex),
                "best_epoch": int(best_epoch),
                "wait": int(wait),
                "improved": int(improved),
                "computed_cindex": int(computed_cindex),
                "epoch_seconds": float(epoch_seconds),
                "elapsed_seconds": float(elapsed_seconds),
            }
        )

        should_log_epoch = (
            log_progress
            and (
                epoch == 1
                or epoch == epochs
                or epoch % verbose_every == 0
                or improved
            )
        )
        if should_log_epoch:
            progress_pct = 100.0 * epoch / epochs
            val_cindex_text = (
                f"{last_val_cindex:.4f}" if not np.isnan(last_val_cindex) else "nan"
            )
            train_cindex_text = (
                f"{last_train_cindex:.4f}" if not np.isnan(last_train_cindex) else "skipped"
            )
            print(
                f"[{epoch:4d}/{epochs}] "
                f"progress={progress_pct:6.2f}% "
                f"train_loss={train_loss.item():.4f} "
                f"val_loss={val_loss_value:.4f} "
                f"train_cindex={train_cindex_text} "
                f"val_cindex={val_cindex_text} "
                f"best_val_loss={best_val_loss:.4f} "
                f"wait={wait}/{patience} "
                f"epoch_time={epoch_seconds:.2f}s "
                f"elapsed={elapsed_seconds:.2f}s"
            )
            if improved and selection_metric == "loss":
                print(
                    f"  New best validation loss at epoch {epoch}: {best_val_loss:.4f}"
                )
            if improved and selection_metric == "cindex":
                print(
                    f"  New best validation C-index at epoch {epoch}: {best_val_cindex:.4f}"
                )
            if computed_cindex and not np.isnan(last_val_cindex):
                print(
                    f"  Validation C-index checkpoint at epoch {epoch}: {last_val_cindex:.4f} "
                    f"(best observed: {best_val_cindex:.4f})"
                )

        should_stop = selection_metric == "loss" or computed_cindex
        if should_stop and wait >= patience:
            if log_progress:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best checkpoint epoch: {best_epoch}. "
                    f"Best validation loss: {best_val_loss:.4f}. "
                    f"Best validation C-index: {best_val_cindex:.4f}. "
                    f"Total elapsed time: {elapsed_seconds:.2f}s."
                )
            break

    model.load_state_dict(best_state)
    if log_progress and history:
        total_elapsed = time.perf_counter() - training_start
        print(
            f"Finished Deep Cox training after {len(history)} epochs. "
            f"Best checkpoint epoch: {best_epoch}. "
            f"Best validation loss: {best_val_loss:.4f}. "
            f"Best observed validation C-index: {best_val_cindex:.4f}. "
            f"Total elapsed time: {total_elapsed:.2f}s."
        )
    return model, pd.DataFrame(history)


def estimate_breslow_baseline_hazard(
    time: Sequence[float],
    event: Sequence[float],
    log_risk: Sequence[float],
) -> pd.DataFrame:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    log_risk = np.asarray(log_risk, dtype=float)
    hazard_ratio = np.exp(log_risk)

    event_times = np.sort(np.unique(time[event == 1]))
    if len(event_times) == 0:
        raise ValueError("At least one observed event is required.")

    increments: list[float] = []
    for current_time in event_times:
        event_count = float(event[(time == current_time)].sum())
        risk_set_sum = float(hazard_ratio[time >= current_time].sum())
        increments.append(event_count / risk_set_sum)

    baseline = pd.DataFrame(
        {
            "time": event_times,
            "baseline_hazard": increments,
        }
    )
    baseline["baseline_cumulative_hazard"] = baseline["baseline_hazard"].cumsum()
    baseline["baseline_survival"] = np.exp(-baseline["baseline_cumulative_hazard"])
    return baseline


def predict_survival_curves(
    baseline_hazard: pd.DataFrame,
    log_risk: Sequence[float],
    index: Sequence[int] | None = None,
) -> pd.DataFrame:
    risk_multiplier = np.exp(np.asarray(log_risk, dtype=float))
    survival_columns: dict[str, np.ndarray] = {"time": baseline_hazard["time"].to_numpy()}

    if index is None:
        index = range(len(risk_multiplier))

    cumulative_hazard = baseline_hazard["baseline_cumulative_hazard"].to_numpy()
    for sample_index in index:
        sample_survival = np.exp(-cumulative_hazard * risk_multiplier[sample_index])
        survival_columns[f"sample_{sample_index}"] = sample_survival

    return pd.DataFrame(survival_columns)
