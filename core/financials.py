from typing import Dict, List, Optional
import math
import pandas as pd

# REQUIRED_FIELDS and _validate_inputs assumed same as before
REQUIRED_FIELDS = {
    "mrr",
    "monthly_growth_pct",
    "churn_pct",
    "arpu",
    "cac",
    "fixed_costs",
    "variable_costs_pct",
    "payment_lag_days",
    "horizon_months",
}


def _validate_inputs(inputs: Dict) -> None:
    missing = REQUIRED_FIELDS - set(inputs.keys())
    if missing:
        raise ValueError(f"Missing required inputs: {', '.join(sorted(missing))}")

    # basic checks (as in previous version)
    def num(key):
        return inputs.get(key)

    for k in ["mrr", "arpu", "cac", "fixed_costs"]:
        v = num(k)
        if not isinstance(v, (int, float)):
            raise ValueError(f"{k} must be a number")
        if v < 0:
            raise ValueError(f"{k} must be >= 0")

    for k in ["monthly_growth_pct", "churn_pct", "variable_costs_pct"]:
        v = num(k)
        if not isinstance(v, (int, float)):
            raise ValueError(f"{k} must be a number")
        if v < 0:
            raise ValueError(f"{k} must be >= 0")

    if inputs["variable_costs_pct"] > 100:
        raise ValueError("variable_costs_pct must be <= 100")
    if inputs["churn_pct"] < 0 or inputs["churn_pct"] > 100:
        raise ValueError("churn_pct must be between 0 and 100")
    if inputs["monthly_growth_pct"] < 0:
        raise ValueError("monthly_growth_pct must be >= 0")

    lag = inputs.get("payment_lag_days")
    if not isinstance(lag, int):
        if isinstance(lag, float) and lag.is_integer():
            inputs["payment_lag_days"] = int(lag)
        else:
            raise ValueError("payment_lag_days must be integer days")
    if inputs["payment_lag_days"] < 0 or inputs["payment_lag_days"] > 365:
        raise ValueError("payment_lag_days must be 0..365")

    h = inputs.get("horizon_months")
    if not isinstance(h, int):
        if isinstance(h, float) and h.is_integer():
            inputs["horizon_months"] = int(h)
        else:
            raise ValueError("horizon_months must be an integer")
    if inputs["horizon_months"] < 1 or inputs["horizon_months"] > 120:
        raise ValueError("horizon_months must be between 1 and 120")


def calculate_pnl(inputs: Dict) -> pd.DataFrame:
    """
    Returns P&L DataFrame for months 1..horizon_months.
    Vectorized implementation.
    """
    _validate_inputs(inputs)

    mrr = float(inputs["mrr"])
    growth_rate = float(inputs["monthly_growth_pct"]) / 100.0
    variable_pct = float(inputs["variable_costs_pct"]) / 100.0
    fixed = float(inputs["fixed_costs"])
    horizon = int(inputs["horizon_months"])

    months = pd.RangeIndex(start=1, stop=horizon + 1, name="month")

    # Vectorized growth factors as a numpy/pandas array, then explicit Series
    growth_factors = (1 + growth_rate) ** (months - 1)
    revenue_series = pd.Series(mrr * growth_factors, index=months, name="revenue")

    cogs = revenue_series * variable_pct
    gross_profit = revenue_series - cogs
    fixed_costs = pd.Series(fixed, index=months, name="fixed_costs")
    operating_profit = gross_profit - fixed_costs
    net_profit = operating_profit.copy()

    df = pd.DataFrame(
        {
            "revenue": revenue_series,
            "cogs": cogs,
            "gross_profit": gross_profit,
            "fixed_costs": fixed_costs,
            "operating_profit": operating_profit,
            "net_profit": net_profit,
        }
    )
    df.index.name = "month"
    return df


def calculate_cashflow(inputs: Dict) -> pd.DataFrame:
    """
    Returns cashflow DataFrame.
    Uses pandas.Series.shift for receipts calculation.
    """
    _validate_inputs(inputs)
    pnl = calculate_pnl(inputs)
    lag_days = int(inputs["payment_lag_days"])
    lag_months = int(math.ceil(lag_days / 30.0)) if lag_days > 0 else 0

    # shift revenue forward by lag_months so revenue at month t is received at month t+lag_months
    receipts = pnl["revenue"].shift(lag_months, fill_value=0.0)
    receipts.name = "receipts"

    payments = pnl["cogs"] + pnl["fixed_costs"]
    net_cash = receipts - payments
    cumulative_cash = net_cash.cumsum()

    df = pd.DataFrame(
        {
            "receipts": receipts,
            "payments": payments,
            "net_cash": net_cash,
            "cumulative_cash": cumulative_cash,
        }
    )
    df.index.name = "month"
    return df


def calculate_unit_economics(inputs: Dict) -> Dict:
    """
    Returns dict with unit economics: ltv, ltv_cac, break_even_month, warnings
    """
    _validate_inputs(inputs)
    arpu = float(inputs["arpu"])
    churn_pct = float(inputs["churn_pct"])
    cac = float(inputs["cac"])
    warnings: List[str] = []

    if churn_pct <= 0:
        raise ValueError("churn_pct must be > 0 to calculate LTV")
    ltv = arpu / (churn_pct / 100.0)

    ltv_cac = ltv / cac if cac > 0 else None
    if cac <= 0:
        warnings.append("CAC is 0 or negative — LTV/CAC undefined")

    cashflow = calculate_cashflow(inputs)
    positive_cashflow = cashflow["cumulative_cash"] >= 0

    bem: Optional[int] = None
    if positive_cashflow.any():
        bem = int(positive_cashflow.idxmax())

    if inputs.get("payment_lag_days", 0) > 60:
        warnings.append("HIGH_PAYMENT_LAG")

    return {
        "ltv": ltv,
        "ltv_cac": ltv_cac,
        "break_even_month": bem,
        "warnings": warnings,
    }
