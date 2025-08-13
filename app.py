# app.py
import json
from copy import deepcopy
from typing import Dict, Any

import streamlit as st
from jsonschema import validate, ValidationError

import pandas as pd

from core.financials import (
    calculate_pnl,
    calculate_cashflow,
    calculate_unit_economics,
)

# ---------------------------
# JSON schema (из Документа D, минимальная проверка)
# ---------------------------
INPUTS_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ScenarioInputs",
    "type": "object",
    "properties": {
        "mrr": {"type": "number", "minimum": 0},
        "monthly_growth_pct": {"type": "number", "minimum": 0, "maximum": 1000000},
        "churn_pct": {"type": "number", "minimum": 0, "maximum": 100},
        "arpu": {"type": "number", "minimum": 0},
        "cac": {"type": "number", "minimum": 0},
        "fixed_costs": {"type": "number", "minimum": 0},
        "variable_costs_pct": {"type": "number", "minimum": 0, "maximum": 100},
        "payment_lag_days": {"type": "integer", "minimum": 0, "maximum": 365},
        "horizon_months": {"type": "integer", "minimum": 1, "maximum": 120},
    },
    "required": [
        "mrr",
        "monthly_growth_pct",
        "churn_pct",
        "arpu",
        "cac",
        "fixed_costs",
        "variable_costs_pct",
    ],
}

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Scenario — Financial Simulator", layout="wide")
st.title("Scenario — Financial Simulator")

# ---------------------------
# Sidebar inputs (все ключевые поля из ТЗ)
# ---------------------------
with st.sidebar:
    st.header("Inputs")

    mrr = st.number_input("MRR (USD / month)", min_value=0.0, value=5000.0, step=100.0, format="%.2f")
    monthly_growth_pct = st.number_input(
        "Monthly growth (%)", min_value=0.0, max_value=1000000.0, value=5.0, step=0.1, format="%.3f"
    )
    churn_pct = st.number_input("Churn (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.1, format="%.3f")
    arpu = st.number_input("ARPU (USD / user / month)", min_value=0.0, value=50.0, step=1.0, format="%.2f")
    cac = st.number_input("CAC (USD)", min_value=0.0, value=200.0, step=10.0, format="%.2f")
    fixed_costs = st.number_input("Fixed costs (USD / month)", min_value=0.0, value=3000.0, step=100.0, format="%.2f")
    variable_costs_pct = st.number_input(
        "Variable costs (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.1, format="%.3f"
    )
    payment_lag_days = st.number_input("Payment lag (days)", min_value=0, max_value=365, value=30, step=1)
    horizon_months = st.number_input("Horizon (months)", min_value=1, max_value=120, value=12, step=1)

    st.markdown("---")
    allow_send_ai = st.checkbox("Разрешаю отправку данных в AI (PII не отправляется по умолчанию)", value=False)

# Build inputs dict
inputs: Dict[str, Any] = {
    "mrr": float(mrr),
    "monthly_growth_pct": float(monthly_growth_pct),
    "churn_pct": float(churn_pct),
    "arpu": float(arpu),
    "cac": float(cac),
    "fixed_costs": float(fixed_costs),
    "variable_costs_pct": float(variable_costs_pct),
    "payment_lag_days": int(payment_lag_days),
    "horizon_months": int(horizon_months),
}

# ---------------------------
# Validation helper
# ---------------------------
def validate_inputs_schema(inp: Dict) -> None:
    """Валидация по JSON Schema. Бросает ValidationError при ошибке."""
    validate(instance=inp, schema=INPUTS_SCHEMA)


# ---------------------------
# Cached calculation wrapper
# ---------------------------
@st.cache_data(ttl=300)
def run_calculations(inp: Dict) -> Dict[str, Any]:
    """Обёртка для кэширования тяжёлых вычислений. Возвращает pnl, cashflow, unit_metrics, warnings."""
    # deepcopy чтобы не мутировать внешний объект
    local = deepcopy(inp)

    # validate schema (jsonschema ValidationError -> handled by caller)
    validate_inputs_schema(local)

    pnl_df = calculate_pnl(local)
    cf_df = calculate_cashflow(local)

    # unit economics может выбросить ValueError при churn==0 в зависимости от реализации.
    # Обработаем это на уровне UI (чтобы не падать).
    try:
        unit_metrics = calculate_unit_economics(local)
    except ValueError as e:
        # если churn == 0 — формируем предупреждение и возвращаем частичные метрики
        err_msg = str(e)
        unit_metrics = {"ltv": None, "ltv_cac": None, "break_even_month": None, "warnings": [err_msg]}

    return {"pnl": pnl_df, "cashflow": cf_df, "unit_metrics": unit_metrics}


# ---------------------------
# Run & display
# ---------------------------
# Unified error object template (B.6)
def error_obj(code: str, message: str) -> dict:
    return {"status": "error", "code": code, "message": message}


calc_result = None
try:
    with st.spinner("Вычисляем..."):
        calc_result = run_calculations(inputs)
except ValidationError as ve:
    st.error("Input Error: Некорректные входные данные (см. подробности).")
    st.json(error_obj("INVALID_INPUT", str(ve)))
except Exception as e:
    # catch-all: показываем унифицированную ошибку и traceback в expander для разработчика
    st.error("Произошла ошибка выполнения.")
    st.json(error_obj("INTERNAL_ERROR", str(e)))
    st.expander("Traceback (debug)").write(e)

if calc_result:
    pnl_df: pd.DataFrame = calc_result["pnl"]
    cf_df: pd.DataFrame = calc_result["cashflow"]
    metrics = calc_result["unit_metrics"]

    # Show warnings (если есть)
    warnings = metrics.get("warnings", [])
    if warnings:
        for w in warnings:
            st.warning(w)

    # Layout: Unit Economics + tabs
    left, right = st.columns([1, 2])

    with left:
        st.header("Unit Economics")
        ltv = metrics.get("ltv")
        ltv_str = f"${ltv:,.2f}" if (ltv is not None) else "—"
        ltv_cac = metrics.get("ltv_cac")
        ltv_cac_str = f"{ltv_cac:.2f}" if (ltv_cac is not None) else "—"
        bem = metrics.get("break_even_month")

        st.metric("LTV", ltv_str)
        st.metric("LTV / CAC", ltv_cac_str)
        st.metric("Break-even month", bem if bem is not None else "—")

        st.markdown("---")
        # AI interpreter stub
        if st.button("Получить рекомендацию (AI)"):
            if not allow_send_ai:
                st.info("Чтобы использовать AI-интерпретатор, отметьте чекбокс 'Разрешаю отправку данных в AI' в сайдбаре.")
            else:
                with st.spinner("Запрошено у AI..."):
                    # Заглушка: здесь вы бы вызвали API (и не отправляйте PII)
                    # Пример fallback:
                    st.success("AI: Сократите CAC и улучшите удержание — это повысит LTV/CAC.")
                    st.info("Примечание: это демонстрационная рекомендация (заглушка).")

    with right:
        st.header("Отчёты")
        tabs = st.tabs(["P&L", "Cash Flow", "Charts", "Export", "How it's calculated"])

        with tabs[0]:
            st.subheader("P&L")
            st.dataframe(pnl_df, use_container_width=True)

        with tabs[1]:
            st.subheader("Cash Flow")
            st.dataframe(cf_df, use_container_width=True)

        with tabs[2]:
            st.subheader("Charts")
            st.line_chart(pnl_df["revenue"], height=250)
            st.line_chart(pnl_df["gross_profit"], height=200)
            st.line_chart(cf_df[["receipts", "payments"]], height=250)

        with tabs[3]:
            st.subheader("Export")
            csv_pnl = pnl_df.to_csv().encode("utf-8")
            csv_cf = cf_df.to_csv().encode("utf-8")
            json_inputs = json.dumps(inputs, indent=2)

            st.download_button("Скачать P&L CSV", csv_pnl, file_name="pnl.csv", mime="text/csv")
            st.download_button("Скачать CashFlow CSV", csv_cf, file_name="cashflow.csv", mime="text/csv")
            st.download_button("Скачать inputs JSON", json_inputs, file_name="inputs.json", mime="application/json")

        with tabs[4]:
            st.subheader("How it's calculated")
            st.write(
                r"""
- LTV = ARPU / (Churn Rate / 100). Если churn = 0 → нестабильно для LTV.
- Revenue моделируется как MRR * (1 + monthly_growth)^(t-1).
- COGS = revenue * variable_costs_pct.
- Receipts учитывают payment_lag_days (см. документацию).
"""
            )
            st.latex(r"\mathrm{LTV} = \frac{ARPU}{Churn\_rate / 100}")

    # Small footer / diagnostics
    st.sidebar.markdown("---")
    st.sidebar.write("Diagnostics")
    st.sidebar.write(f"Cached: run_calculations cache TTL = 300s")
