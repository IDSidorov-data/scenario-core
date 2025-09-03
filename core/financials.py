# financials.py
"""
Core financial calculation engine for the scenario planner.

This module contains functions to calculate P&L, cash flow, and key
unit economic metrics based on a set of user-provided inputs.
"""

from typing import Dict, List, Optional
import math
import numpy as np
import pandas as pd

# A set of required input fields for all financial calculations.
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
    """Validates the user input dictionary.

    Ensures that all required fields are present and that their values
    are of the correct type and within acceptable ranges.

    Args:
        inputs: A dictionary containing user-provided financial parameters.

    Raises:
        ValueError: If any input is missing, has the wrong type,
                    or is outside its allowed range.
    """
    missing = REQUIRED_FIELDS - set(inputs.keys())
    if missing:
        raise ValueError(f"Missing required inputs: {', '.join(sorted(missing))}")

    for k in ["mrr", "arpu", "cac", "fixed_costs", "monthly_growth_pct"]:
        v = inputs.get(k)
        if not isinstance(v, (int, float)) or v < 0:
            raise ValueError(f"{k} must be a non-negative number")

    for k in ["churn_pct", "variable_costs_pct"]:
        v = inputs.get(k)
        if not isinstance(v, (int, float)) or not 0 <= v <= 100:
            raise ValueError(f"{k} must be between 0 and 100")

    for key in ["payment_lag_days", "horizon_months"]:
        val = inputs.get(key)
        if not isinstance(val, int):
            # Allow floats that are whole numbers (e.g., 12.0)
            if isinstance(val, float) and val.is_integer():
                inputs[key] = int(val)
            else:
                raise ValueError(f"{key} must be an integer")

    if not 0 <= inputs["payment_lag_days"] <= 365:
        raise ValueError("payment_lag_days must be 0..365")
    if not 1 <= inputs["horizon_months"] <= 120:
        raise ValueError("horizon_months must be between 1 and 120")


def calculate_pnl(inputs: Dict) -> pd.DataFrame:
    """Calculates the Profit and Loss (P&L) statement over a given time horizon.

    This function projects revenue based on initial MRR and a monthly growth rate.
    It then calculates COGS, gross profit, and operating/net profit.

    Args:
        inputs: The validated dictionary of financial parameters.

    Returns:
        A pandas DataFrame with P&L items (revenue, cogs, etc.) indexed by month.
    """
    _validate_inputs(inputs)
    mrr, growth_rate, variable_pct, fixed, horizon = (
        float(inputs["mrr"]),
        float(inputs["monthly_growth_pct"]) / 100.0,
        float(inputs["variable_costs_pct"]) / 100.0,
        float(inputs["fixed_costs"]),
        int(inputs["horizon_months"]),
    )
    months = pd.RangeIndex(start=1, stop=horizon + 1, name="month")

    # Project revenue growth using a compound growth formula.
    growth_factors = (1 + growth_rate) ** (months - 1)
    revenue_series = pd.Series(mrr * growth_factors, index=months, name="revenue")

    cogs = revenue_series * variable_pct
    gross_profit = revenue_series - cogs
    fixed_costs = pd.Series(fixed, index=months, name="fixed_costs")
    operating_profit = gross_profit - fixed_costs

    # In this model, net profit equals operating profit (no taxes/interest).
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


def _compute_receipts_from_revenue(
    revenue_series: pd.Series, lag_days: int, horizon: int
) -> pd.Series:
    """Distributes monthly revenue into actual cash receipts based on a payment lag.

    This helper function models the delay between when revenue is earned and
    when cash is received. For example, a 15-day lag means half of a month's
    revenue is received in that month, and half in the next.

    Args:
        revenue_series: A Series of monthly revenues.
        lag_days: The average number of days it takes to receive payment.
        horizon: The total number of months in the forecast.

    Returns:
        A Series of monthly cash receipts.
    """
    receipts = np.zeros(horizon)
    # Assume an average of 30 days per month for simplicity.
    lag_months_float = lag_days / 30.0

    for t, revenue in enumerate(revenue_series.values):
        # Calculate the floating-point month index when cash is received.
        receive_at = t + lag_months_float
        k = math.floor(receive_at)  # The integer part of the month index.
        frac = receive_at - k  # The fractional part for splitting revenue.

        # Allocate the first portion of the revenue to month k.
        if k < horizon:
            receipts[k] += revenue * (1 - frac)

        # Allocate the second portion to the next month, k+1.
        if k + 1 < horizon:
            receipts[k + 1] += revenue * frac

    return pd.Series(receipts, index=pd.RangeIndex(start=1, stop=horizon + 1))


def calculate_cashflow(inputs: Dict) -> pd.DataFrame:
    """Calculates the monthly and cumulative cash flow statement.

    First, it calculates the P&L to get revenues and costs. Then, it adjusts
    revenues for payment lag to get actual receipts. Finally, it computes

    net cash flow and cumulative cash position over the horizon.

    Args:
        inputs: The validated dictionary of financial parameters.

    Returns:
        A pandas DataFrame with cash flow items (receipts, payments, etc.)
        indexed by month.
    """
    _validate_inputs(inputs)
    pnl = calculate_pnl(inputs)
    lag_days = int(inputs["payment_lag_days"])
    horizon = int(inputs["horizon_months"])

    receipts = _compute_receipts_from_revenue(pnl["revenue"], lag_days, horizon)
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
    """Calculates key unit economic metrics.

    Computes Lifetime Value (LTV), LTV-to-CAC ratio, and the break-even month (BEM).
    It also generates warnings for potentially problematic input values.

    Args:
        inputs: The validated dictionary of financial parameters.

    Returns:
        A dictionary containing the calculated metrics ('ltv', 'ltv_cac',
        'break_even_month') and a list of 'warnings'.
    """
    _validate_inputs(inputs)
    arpu, churn_pct, cac = (
        float(inputs["arpu"]),
        float(inputs["churn_pct"]),
        float(inputs["cac"]),
    )
    warnings: List[str] = []

    # LTV is undefined (infinite) if churn is zero.
    ltv = arpu / (churn_pct / 100.0) if churn_pct > 0 else None
    if churn_pct == 0:
        warnings.append("CHURN_ZERO")

    # LTV/CAC ratio is undefined if CAC is zero.
    ltv_cac = ltv / cac if cac > 0 and ltv is not None else None
    if cac <= 0:
        warnings.append("CAC_ZERO")

    cashflow = calculate_cashflow(inputs)
    positive_cashflow = cashflow["cumulative_cash"] >= 0

    # Find the first month where cumulative cash is non-negative.
    bem: Optional[int] = (
        int(positive_cashflow.idxmax()) if positive_cashflow.any() else None
    )

    if inputs.get("payment_lag_days", 0) > 60:
        warnings.append("HIGH_PAYMENT_LAG")

    return {
        "ltv": ltv,
        "ltv_cac": ltv_cac,
        "break_even_month": bem,
        "warnings": warnings,
    }
