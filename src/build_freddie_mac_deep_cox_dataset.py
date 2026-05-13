from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_FEATURE_COLUMNS = [
    "CreditScore",
    "FirstTimeHomebuyerFlag",
    "MortgageInsurancePercentage",
    "NumberofUnits",
    "OccupancyStatus",
    "OriginalCombinedLoantoValueCLTV",
    "OriginalDebttoIncomeRatio",
    "OriginalUPB",
    "OriginalLoantoValueLTV",
    "OriginalInterestRate",
    "Channel",
    "PropertyState",
    "PropertyType",
    "LoanPurpose",
    "OriginalLoanTerm",
    "NumberofBorrowers",
]

NUMERIC_SENTINELS = {
    "CreditScore": {9999},
    "MortgageInsurancePercentage": {999},
    "NumberofUnits": {99},
    "OriginalCombinedLoantoValueCLTV": {999},
    "OriginalDebttoIncomeRatio": {999},
    "OriginalLoantoValueLTV": {999},
    "OriginalLoanTerm": {-8, 0},
    "NumberofBorrowers": {99},
}

CATEGORICAL_SENTINELS = {"", "9", "99", "999", "9999"}
CATEGORICAL_COLUMNS = [
    "FirstTimeHomebuyerFlag",
    "OccupancyStatus",
    "Channel",
    "PropertyState",
    "PropertyType",
    "LoanPurpose",
]

# Freddie Mac ZeroBalanceCode 01 = "Prepaid or Matured (Voluntary Payoff)".
# We treat all other zero-balance outcomes as competing terminations and censor
# them at their termination month for the Part D prepayment model.
PREPAYMENT_ZERO_BALANCE_CODES = {1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Freddie Mac loan-level survival dataset for the Deep Cox notebook."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2000,
        help="Origination/performance year to process. Default: 2000.",
    )
    parser.add_argument(
        "--orig-dir",
        type=Path,
        default=Path("Freddie_Mac_Loan_Data/Origination_Historical_Data"),
        help="Directory containing historical_data_<year>.parquet files.",
    )
    parser.add_argument(
        "--perf-dir",
        type=Path,
        default=Path("Freddie_Mac_Loan_Data/Monthly_Performance_historical_data_time"),
        help="Directory containing historical_data_time_<year>.parquet files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path. Defaults to Freddie_Mac_Loan_Data/deep_cox_<year>.parquet.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=250_000,
        help="Number of monthly performance rows to process per batch.",
    )
    return parser.parse_args()


def load_origination_data(path: Path, year: int) -> pd.DataFrame:
    columns = ["LoanSequenceNumber", *DEFAULT_FEATURE_COLUMNS]
    df = pd.read_parquet(path, columns=columns)
    df["LoanSequenceNumber"] = df["LoanSequenceNumber"].astype("string")

    for column, sentinels in NUMERIC_SENTINELS.items():
        if column not in df.columns:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].mask(df[column].isin(sentinels))

    if "OriginalInterestRate" in df.columns:
        df["OriginalInterestRate"] = pd.to_numeric(df["OriginalInterestRate"], errors="coerce")
        df["OriginalInterestRate"] = df["OriginalInterestRate"].mask(
            df["OriginalInterestRate"] <= 0
        )

    if "OriginalUPB" in df.columns:
        df["OriginalUPB"] = pd.to_numeric(df["OriginalUPB"], errors="coerce")
        df["OriginalUPB"] = df["OriginalUPB"].mask(df["OriginalUPB"] <= 0)

    for column in CATEGORICAL_COLUMNS:
        if column not in df.columns:
            continue
        series = df[column].astype("string").str.strip()
        df[column] = series.mask(series.isin(CATEGORICAL_SENTINELS))

    df["origination_year"] = year
    return df


def summarize_performance_data(path: Path, batch_size: int) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(path)
    columns = [
        "LoanSequenceNumber",
        "MonthlyReportingPeriod",
        "LoanAge",
        "ZeroBalanceCode",
    ]

    max_loan_age: dict[str, int] = {}
    last_reporting_period: dict[str, int] = {}
    first_event_age: dict[str, int] = {}
    first_event_period: dict[str, int] = {}
    first_event_code: dict[str, int] = {}

    rows_processed = 0
    for batch_number, batch in enumerate(
        parquet_file.iter_batches(columns=columns, batch_size=batch_size),
        start=1,
    ):
        batch_df = batch.to_pandas()
        rows_processed += len(batch_df)

        batch_df["LoanSequenceNumber"] = batch_df["LoanSequenceNumber"].astype("string")
        batch_df["MonthlyReportingPeriod"] = pd.to_numeric(
            batch_df["MonthlyReportingPeriod"],
            errors="coerce",
        )
        batch_df["LoanAge"] = pd.to_numeric(batch_df["LoanAge"], errors="coerce")
        batch_df["ZeroBalanceCode"] = pd.to_numeric(batch_df["ZeroBalanceCode"], errors="coerce")

        censored_batch = (
            batch_df.groupby("LoanSequenceNumber", as_index=False, observed=True)
            .agg(
                max_loan_age=("LoanAge", "max"),
                last_reporting_period=("MonthlyReportingPeriod", "max"),
            )
        )
        for row in censored_batch.itertuples(index=False):
            loan = str(row.LoanSequenceNumber)
            batch_max_age = int(row.max_loan_age)
            batch_last_period = int(row.last_reporting_period)

            previous_max_age = max_loan_age.get(loan)
            if previous_max_age is None or batch_max_age > previous_max_age:
                max_loan_age[loan] = batch_max_age

            previous_last_period = last_reporting_period.get(loan)
            if previous_last_period is None or batch_last_period > previous_last_period:
                last_reporting_period[loan] = batch_last_period

        termination_rows = batch_df[batch_df["ZeroBalanceCode"].notna()].copy()
        if not termination_rows.empty:
            termination_rows = termination_rows.sort_values(
                ["LoanSequenceNumber", "LoanAge", "MonthlyReportingPeriod"]
            )
            termination_batch = (
                termination_rows.groupby("LoanSequenceNumber", as_index=False, observed=True)
                .first()
            )
            for row in termination_batch.itertuples(index=False):
                loan = str(row.LoanSequenceNumber)
                event_age = int(row.LoanAge)
                event_period = int(row.MonthlyReportingPeriod)
                event_code = int(row.ZeroBalanceCode)

                previous_age = first_event_age.get(loan)
                previous_period = first_event_period.get(loan)
                should_replace = (
                    previous_age is None
                    or event_age < previous_age
                    or (event_age == previous_age and event_period < previous_period)
                )
                if should_replace:
                    first_event_age[loan] = event_age
                    first_event_period[loan] = event_period
                    first_event_code[loan] = event_code

        if batch_number == 1 or batch_number % 25 == 0:
            print(
                f"Processed batch {batch_number} "
                f"({rows_processed:,} monthly rows so far)."
            )

    performance_summary = pd.DataFrame(
        {
            "LoanSequenceNumber": list(max_loan_age.keys()),
            "last_observed_loan_age": list(max_loan_age.values()),
            "last_reporting_period": [last_reporting_period[key] for key in max_loan_age],
        }
    )

    performance_summary["event"] = (
        performance_summary["LoanSequenceNumber"]
        .map(first_event_code)
        .isin(PREPAYMENT_ZERO_BALANCE_CODES)
        .astype(np.int8)
    )
    performance_summary["event_loan_age"] = performance_summary["LoanSequenceNumber"].map(
        first_event_age
    )
    performance_summary["event_reporting_period"] = performance_summary[
        "LoanSequenceNumber"
    ].map(first_event_period)
    performance_summary["zero_balance_code"] = performance_summary[
        "LoanSequenceNumber"
    ].map(first_event_code)

    # Any zero-balance event ends the loan's observable life. Only voluntary
    # payoff counts as a positive event; all other termination codes are
    # censored at the termination month.
    performance_summary["duration_months"] = np.where(
        performance_summary["event_loan_age"].notna(),
        performance_summary["event_loan_age"],
        performance_summary["last_observed_loan_age"],
    )

    performance_summary["duration_months"] = performance_summary["duration_months"].astype(
        "int32"
    )
    performance_summary["last_observed_loan_age"] = performance_summary[
        "last_observed_loan_age"
    ].astype("int32")

    return performance_summary


def build_dataset(year: int, orig_dir: Path, perf_dir: Path, batch_size: int) -> pd.DataFrame:
    orig_path = orig_dir / f"historical_data_{year}.parquet"
    perf_path = perf_dir / f"historical_data_time_{year}.parquet"

    if not orig_path.exists():
        raise FileNotFoundError(f"Missing origination file: {orig_path}")
    if not perf_path.exists():
        raise FileNotFoundError(f"Missing performance file: {perf_path}")

    print(f"Loading origination data from {orig_path}")
    origination = load_origination_data(orig_path, year=year)
    print(f"Loaded {len(origination):,} origination rows.")

    print(f"Summarizing performance data from {perf_path}")
    performance = summarize_performance_data(perf_path, batch_size=batch_size)
    print(f"Summarized {len(performance):,} loan-level performance rows.")

    dataset = origination.merge(
        performance,
        on="LoanSequenceNumber",
        how="inner",
        validate="one_to_one",
    )

    dataset = dataset.dropna(subset=["duration_months", "event"])
    dataset["event"] = dataset["event"].astype(np.int8)
    dataset["duration_months"] = dataset["duration_months"].astype(np.int32)

    ordered_columns = [
        "LoanSequenceNumber",
        "duration_months",
        "event",
        "zero_balance_code",
        "event_loan_age",
        "event_reporting_period",
        "last_observed_loan_age",
        "last_reporting_period",
        "origination_year",
        *DEFAULT_FEATURE_COLUMNS,
    ]
    existing_order = [column for column in ordered_columns if column in dataset.columns]
    remaining = [column for column in dataset.columns if column not in existing_order]
    return dataset[existing_order + remaining]


def main() -> None:
    args = parse_args()
    output_path = (
        args.output
        if args.output is not None
        else Path(f"Freddie_Mac_Loan_Data/deep_cox_{args.year}.parquet")
    )

    dataset = build_dataset(
        year=args.year,
        orig_dir=args.orig_dir,
        perf_dir=args.perf_dir,
        batch_size=args.batch_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)

    print()
    print(f"Saved Deep Cox dataset to {output_path.resolve()}")
    print(f"Rows: {len(dataset):,}")
    print(f"Columns: {len(dataset.columns):,}")
    print(f"Event rate: {dataset['event'].mean():.4f}")
    print("First columns:")
    print(dataset.columns[:15].tolist())


if __name__ == "__main__":
    main()
