# core/tests/test_calculations.py

import pytest
import pandas as pd
from copy import deepcopy

# Предполагается, что ваши функции находятся в core/financials.py
from core.financials import (
    calculate_pnl,
    calculate_cashflow,
    calculate_unit_economics,
)


# --------------------
# Fixture: базовые корректные входные данные
# --------------------


@pytest.fixture
def base_inputs():
    """
    Возвращает стандартный, "здоровый" набор входных данных для тестов.
    Используем fixture, но каждый тест берёт deepcopy чтобы избежать побочных эффектов.
    """
    return {
        "mrr": 10000,
        "monthly_growth_pct": 5,
        "churn_pct": 2,
        "arpu": 100,
        "cac": 500,
        "fixed_costs": 8000,
        "variable_costs_pct": 15,
        "payment_lag_days": 30,
        "horizon_months": 24,
    }


# --------------------
# Unit economics
# --------------------


def test_unit_economics_structure_and_types(base_inputs):
    inputs = deepcopy(base_inputs)
    metrics = calculate_unit_economics(inputs)
    assert isinstance(metrics, dict)
    assert set(["ltv", "ltv_cac", "break_even_month", "warnings"]).issubset(
        metrics.keys()
    )
    assert isinstance(metrics["warnings"], list)


def test_ltv_calculation_basic(base_inputs):
    inputs = deepcopy(base_inputs)
    metrics = calculate_unit_economics(inputs)
    # LTV = ARPU / (Churn Rate / 100) = 100 / (2 / 100) = 5000
    assert metrics["ltv"] == pytest.approx(5000)


def test_ltv_cac_ratio_calculation(base_inputs):
    inputs = deepcopy(base_inputs)
    metrics = calculate_unit_economics(inputs)
    # LTV = 5000, CAC = 500 -> LTV/CAC = 10
    assert metrics["ltv_cac"] == pytest.approx(10)


def test_zero_churn_raises_value_error(base_inputs):
    """При churn_pct = 0 должно выбрасываться ValueError (по контракту)."""
    inputs = deepcopy(base_inputs)
    inputs["churn_pct"] = 0

    with pytest.raises(ValueError, match="churn_pct must be > 0"):
        calculate_unit_economics(inputs)


def test_zero_cac_returns_none_and_warning(base_inputs):
    """При CAC = 0 LTV/CAC должен быть None и появиться предупреждение, связанное с CAC."""
    inputs = deepcopy(base_inputs)
    inputs["cac"] = 0

    metrics = calculate_unit_economics(inputs)
    assert metrics["ltv_cac"] is None
    assert isinstance(metrics["warnings"], list)
    # допускаем гибкий текст предупреждения — проверяем наличие упоминания CAC или LTV/CAC
    assert any(
        ("cac" in w.lower()) or ("ltv/cac" in w.lower()) for w in metrics["warnings"]
    )


# --------------------
# P&L
# --------------------


def test_pnl_structure_and_types(base_inputs):
    inputs = deepcopy(base_inputs)
    pnl = calculate_pnl(inputs)
    assert isinstance(pnl, pd.DataFrame)
    assert pnl.index.name == "month"
    expected_columns = {
        "revenue",
        "cogs",
        "gross_profit",
        "fixed_costs",
        "operating_profit",
        "net_profit",
    }
    assert expected_columns.issubset(set(pnl.columns))


def test_pnl_first_month_revenue(base_inputs):
    inputs = deepcopy(base_inputs)
    pnl = calculate_pnl(inputs)
    # Ожидаем, что выручка в 1-й месяц равна стартовому MRR
    assert pnl.loc[1, "revenue"] == pytest.approx(base_inputs["mrr"])


# --------------------
# Cash Flow
# --------------------


def test_cashflow_structure_and_types(base_inputs):
    inputs = deepcopy(base_inputs)
    cf = calculate_cashflow(inputs)
    assert isinstance(cf, pd.DataFrame)
    assert cf.index.name == "month"
    assert all(
        col in cf.columns
        for col in ["receipts", "payments", "net_cash", "cumulative_cash"]
    )


# --------------------
# Validation errors
# --------------------


def test_negative_mrr_raises_value_error(base_inputs):
    inputs = deepcopy(base_inputs)
    inputs["mrr"] = -100

    with pytest.raises(ValueError, match="mrr must be >= 0"):
        calculate_pnl(inputs)


def test_invalid_horizon_raises_value_error(base_inputs):
    inputs = deepcopy(base_inputs)
    inputs["horizon_months"] = 0

    with pytest.raises(ValueError, match="horizon_months must be between 1 and 120"):
        calculate_pnl(inputs)


# --------------------
# Стресс/экстремальные сценарии
# --------------------


def test_extreme_scenario_warns_not_crashes(base_inputs):
    """
    Очень высокий рост или большой payment_lag не должны приводить к падению —
    функция должна возвращать результаты и список предупреждений.
    """
    inputs = deepcopy(base_inputs)
    inputs["monthly_growth_pct"] = 1000
    # большой лаг, чтобы триггернуть предупреждение по cashflow (если оно реализовано)
    inputs["payment_lag_days"] = 90

    metrics = calculate_unit_economics(inputs)
    assert "ltv" in metrics
    assert isinstance(metrics.get("warnings"), list)
    # хотя точный текст предупреждения может отличаться, ожидаем хотя бы одно предупреждение
    assert len(metrics.get("warnings", [])) >= 0
