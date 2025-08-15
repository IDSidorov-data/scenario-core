# core/tests/test_calculations.py (ИСПРАВЛЕННАЯ ВЕРСЯ)
import pytest
import pandas as pd
from copy import deepcopy

from core.financials import (
    calculate_pnl,
    calculate_cashflow,
    calculate_unit_economics,
    _validate_inputs,
)

@pytest.fixture
def base_inputs():
    return {
        "mrr": 10000, "monthly_growth_pct": 5, "churn_pct": 2, "arpu": 100,
        "cac": 500, "fixed_costs": 8000, "variable_costs_pct": 15,
        "payment_lag_days": 30, "horizon_months": 24,
    }

def test_unit_economics_structure_and_types(base_inputs):
    metrics = calculate_unit_economics(deepcopy(base_inputs))
    assert isinstance(metrics, dict)
    assert {"ltv", "ltv_cac", "break_even_month", "warnings"}.issubset(metrics.keys())
    assert isinstance(metrics["warnings"], list)

def test_ltv_calculation_basic(base_inputs):
    metrics = calculate_unit_economics(deepcopy(base_inputs))
    assert metrics["ltv"] == pytest.approx(5000)

def test_ltv_cac_ratio_calculation(base_inputs):
    metrics = calculate_unit_economics(deepcopy(base_inputs))
    assert metrics["ltv_cac"] == pytest.approx(10)

# ИЗМЕНЕНО: Тест теперь проверяет корректную обработку, а не падение
def test_zero_churn_returns_none_and_warning(base_inputs):
    """При churn_pct = 0 LTV должен быть None и должно появиться предупреждение CHURN_ZERO."""
    inputs = deepcopy(base_inputs)
    inputs["churn_pct"] = 0
    
    # Валидация должна проходить успешно
    _validate_inputs(inputs)
    
    metrics = calculate_unit_economics(inputs)
    assert metrics["ltv"] is None
    assert "CHURN_ZERO" in metrics["warnings"]

def test_zero_cac_returns_none_and_warning(base_inputs):
    inputs = deepcopy(base_inputs)
    inputs["cac"] = 0
    metrics = calculate_unit_economics(inputs)
    assert metrics["ltv_cac"] is None
    assert "CAC_ZERO" in metrics["warnings"]

def test_pnl_structure_and_types(base_inputs):
    pnl = calculate_pnl(deepcopy(base_inputs))
    assert isinstance(pnl, pd.DataFrame)
    assert pnl.index.name == "month"
    expected_columns = {"revenue", "cogs", "gross_profit", "fixed_costs", "operating_profit", "net_profit"}
    assert expected_columns.issubset(pnl.columns)

def test_pnl_first_month_revenue(base_inputs):
    pnl = calculate_pnl(deepcopy(base_inputs))
    assert pnl.loc[1, "revenue"] == pytest.approx(base_inputs["mrr"])

def test_cashflow_structure_and_types(base_inputs):
    cf = calculate_cashflow(deepcopy(base_inputs))
    assert isinstance(cf, pd.DataFrame)
    assert cf.index.name == "month"
    assert all(col in cf.columns for col in ["receipts", "payments", "net_cash", "cumulative_cash"])

def test_negative_mrr_raises_value_error(base_inputs):
    inputs = deepcopy(base_inputs)
    inputs["mrr"] = -100
    with pytest.raises(ValueError):
        calculate_pnl(inputs)

def test_invalid_horizon_raises_value_error(base_inputs):
    inputs = deepcopy(base_inputs)
    inputs["horizon_months"] = 0
    with pytest.raises(ValueError):
        calculate_pnl(inputs)