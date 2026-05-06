"""
Schemas for Freddie Mac single-family loan-level data conversion.

Defines:
  - ORIG_COLS / MONTHLY_COLS : ordered snake_case column names
  - sentinel maps            : per-column sentinel values that map to null
  - transform_origination()  : raw-string DataFrame -> typed DataFrame
  - transform_monthly()      : raw-string DataFrame -> typed DataFrame

The conversion strategy is to read everything as Utf8, then null-out
sentinels and cast in Polars. This keeps the CSV reader simple and lets
us encode per-column quirks (mixed alpha/numeric fields, varying
sentinel widths) without bespoke per-column parsers.

References:
  data/user_guide.pdf  - Freddie Mac single-family loan-level dataset user guide
  data/file_layout.xlsx - field index/layout
"""

import polars as pl

# 32 origination fields, in raw file order.
ORIG_COLS = [
    "fico",
    "first_payment_date_raw",
    "first_time_homebuyer",
    "maturity_date_raw",
    "msa",
    "mi_pct",
    "n_units",
    "occupancy",
    "cltv",
    "dti",
    "orig_upb",
    "ltv",
    "orig_rate",
    "channel",
    "ppm_flag",
    "amort_type",
    "state",
    "prop_type",
    "zip3",
    "loan_seq_num",
    "loan_purpose",
    "term_months",
    "n_borrowers",
    "seller_name",
    "servicer_name",
    "super_conforming",
    "preharp_loan_seq_num",
    "program",
    "harp",
    "prop_valuation_method",
    "io_indicator",
    "mi_cancel",
]

# 32 monthly performance fields, in raw file order.
MONTHLY_COLS = [
    "loan_seq_num",
    "month_raw",
    "actual_upb",
    "dq_status",
    "loan_age",
    "rem_months",
    "defect_date_raw",
    "mod_flag",
    "zb_code",
    "zb_date_raw",
    "current_rate",
    "deferred_upb",
    "ddlpi_raw",
    "mi_recoveries",
    "net_sales_proceeds",
    "non_mi_recoveries",
    "expenses",
    "legal_costs",
    "maint_costs",
    "taxes_insurance",
    "misc_expenses",
    "actual_loss",
    "mod_cost",
    "step_mod_flag",
    "deferred_payment_plan",
    "eltv",
    "zb_removal_upb",
    "dq_accrued_interest",
    "disaster_dq",
    "borrower_assistance",
    "current_mod_cost",
    "interest_bearing_upb",
]

# Per-column sentinel string values (post empty-string handling).
ORIG_SENTINELS = {
    "fico": ["9999"],
    "first_time_homebuyer": ["9"],
    "mi_pct": ["999"],
    "n_units": ["99"],
    "occupancy": ["9"],
    "cltv": ["999"],
    "dti": ["999"],
    "ltv": ["999"],
    "channel": ["9"],
    "prop_type": ["99"],
    "loan_purpose": ["9"],
    "n_borrowers": ["99"],
    "prop_valuation_method": ["9"],
}

MONTHLY_SENTINELS = {
    "eltv": ["999"],
}


def _yyyymm_to_date(col_name: str) -> pl.Expr:
    """Parse a 'YYYYMM' string column to Date (first of month)."""
    s = pl.col(col_name).cast(pl.Utf8, strict=False).str.strip_chars()
    valid = s.str.len_chars() == 6
    year = pl.when(valid).then(s.str.slice(0, 4).cast(pl.Int32, strict=False)).otherwise(None)
    month = pl.when(valid).then(s.str.slice(4, 2).cast(pl.Int8, strict=False)).otherwise(None)
    return pl.date(year, month, 1)


def _empty_to_null(df: pl.DataFrame) -> pl.DataFrame:
    """Replace empty strings with nulls across all Utf8 columns."""
    string_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Utf8]
    if not string_cols:
        return df
    return df.with_columns(
        [
            pl.when(pl.col(c).str.strip_chars() == "").then(None).otherwise(pl.col(c)).alias(c)
            for c in string_cols
        ]
    )


def _apply_sentinels(df: pl.DataFrame, sentinels: dict[str, list[str]]) -> pl.DataFrame:
    exprs = []
    for col, vals in sentinels.items():
        if col not in df.columns:
            continue
        exprs.append(
            pl.when(pl.col(col).is_in(vals)).then(None).otherwise(pl.col(col)).alias(col)
        )
    if not exprs:
        return df
    return df.with_columns(exprs)


def transform_origination(df: pl.DataFrame, vintage_year: int, vintage_quarter: int) -> pl.DataFrame:
    """Type-cast and clean a raw-string origination frame.

    Adds vintage_year, vintage_quarter columns. Caller is responsible for
    applying the term==360 & amort_type=='FRM' filter (so this function is
    reusable for QC counts of the unfiltered data).
    """
    df = _empty_to_null(df)
    df = _apply_sentinels(df, ORIG_SENTINELS)

    df = df.with_columns(
        [
            _yyyymm_to_date("first_payment_date_raw").alias("first_payment_date"),
            _yyyymm_to_date("maturity_date_raw").alias("maturity_date"),
        ]
    ).drop(["first_payment_date_raw", "maturity_date_raw"])

    df = df.with_columns(
        [
            pl.col("fico").cast(pl.UInt16, strict=False),
            pl.col("msa").cast(pl.UInt32, strict=False),
            pl.col("mi_pct").cast(pl.UInt8, strict=False),
            pl.col("n_units").cast(pl.UInt8, strict=False),
            pl.col("cltv").cast(pl.UInt8, strict=False),
            pl.col("dti").cast(pl.UInt8, strict=False),
            pl.col("orig_upb").cast(pl.Float32, strict=False),
            pl.col("ltv").cast(pl.UInt8, strict=False),
            pl.col("orig_rate").cast(pl.Float32, strict=False),
            pl.col("zip3").cast(pl.UInt32, strict=False),
            pl.col("term_months").cast(pl.UInt16, strict=False),
            pl.col("n_borrowers").cast(pl.UInt8, strict=False),
            pl.col("prop_valuation_method").cast(pl.UInt8, strict=False),
            pl.lit(vintage_year, dtype=pl.UInt16).alias("vintage_year"),
            pl.lit(vintage_quarter, dtype=pl.UInt8).alias("vintage_quarter"),
        ]
    )
    return df


def transform_monthly(df: pl.DataFrame) -> pl.DataFrame:
    """Type-cast and clean a raw-string monthly-performance frame chunk."""
    df = _empty_to_null(df)
    df = _apply_sentinels(df, MONTHLY_SENTINELS)

    df = df.with_columns(
        [
            _yyyymm_to_date("month_raw").alias("month"),
            _yyyymm_to_date("defect_date_raw").alias("defect_date"),
            _yyyymm_to_date("zb_date_raw").alias("zb_date"),
            _yyyymm_to_date("ddlpi_raw").alias("ddlpi"),
        ]
    ).drop(["month_raw", "defect_date_raw", "zb_date_raw", "ddlpi_raw"])

    df = df.with_columns(
        [
            pl.col("actual_upb").cast(pl.Float32, strict=False),
            pl.col("loan_age").cast(pl.UInt16, strict=False),
            pl.col("rem_months").cast(pl.UInt16, strict=False),
            pl.col("current_rate").cast(pl.Float32, strict=False),
            pl.col("deferred_upb").cast(pl.Float32, strict=False),
            pl.col("mi_recoveries").cast(pl.Float32, strict=False),
            pl.col("non_mi_recoveries").cast(pl.Float32, strict=False),
            pl.col("expenses").cast(pl.Float32, strict=False),
            pl.col("legal_costs").cast(pl.Float32, strict=False),
            pl.col("maint_costs").cast(pl.Float32, strict=False),
            pl.col("taxes_insurance").cast(pl.Float32, strict=False),
            pl.col("misc_expenses").cast(pl.Float32, strict=False),
            pl.col("actual_loss").cast(pl.Float32, strict=False),
            pl.col("mod_cost").cast(pl.Float32, strict=False),
            pl.col("eltv").cast(pl.UInt16, strict=False),
            pl.col("zb_removal_upb").cast(pl.Float32, strict=False),
            pl.col("dq_accrued_interest").cast(pl.Float32, strict=False),
            pl.col("current_mod_cost").cast(pl.Float32, strict=False),
            pl.col("interest_bearing_upb").cast(pl.Float32, strict=False),
        ]
    )
    return df


# Final ordered output columns for monthly. Used to lock the parquet schema
# across streamed batches (otherwise pyarrow's ParquetWriter rejects appends).
MONTHLY_OUTPUT_COLS_ORDER = [
    "loan_seq_num",
    "month",
    "actual_upb",
    "dq_status",
    "loan_age",
    "rem_months",
    "defect_date",
    "mod_flag",
    "zb_code",
    "zb_date",
    "current_rate",
    "deferred_upb",
    "ddlpi",
    "mi_recoveries",
    "net_sales_proceeds",
    "non_mi_recoveries",
    "expenses",
    "legal_costs",
    "maint_costs",
    "taxes_insurance",
    "misc_expenses",
    "actual_loss",
    "mod_cost",
    "step_mod_flag",
    "deferred_payment_plan",
    "eltv",
    "zb_removal_upb",
    "dq_accrued_interest",
    "disaster_dq",
    "borrower_assistance",
    "current_mod_cost",
    "interest_bearing_upb",
]
