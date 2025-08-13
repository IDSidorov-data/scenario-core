# app.py

import json
from copy import deepcopy
from typing import Dict, Any
from pathlib import Path

import streamlit as st
from jsonschema import validate, ValidationError
import pandas as pd

# --- core financials (assumed implemented in core/financials.py) ---
from core.financials import (
    calculate_pnl,
    calculate_cashflow,
    calculate_unit_economics,
)

# ---------------------------
# Robust Localization loader
# ---------------------------
@st.cache_data
def load_translation(lang: str = "ru") -> Dict[str, Any]:
    """
    Попытки найти файл локализации в нескольких местах.
    Проверяет в порядке:
      1) <dir_of_this_file>/core/locales/{lang}.json  (если app.py находится в репо корне)
      2) <dir_of_this_file>/locales/{lang}.json
      3) cwd()/core/locales/{lang}.json
      4) cwd()/locales/{lang}.json

    Возвращает словарь или пустой dict.
    """
    fname = f"{lang}.json"
    candidates = []

    # 1) relative to this file's parent (two possibilities: project root or core/)
    try:
        here = Path(__file__).resolve().parent
        candidates.append(here / "core" / "locales" / fname)
        candidates.append(here / "locales" / fname)
    except Exception:
        pass

    # 2) relative to cwd
    cwd = Path.cwd()
    candidates.append(cwd / "core" / "locales" / fname)
    candidates.append(cwd / "locales" / fname)

    for p in candidates:
        try:
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            st.warning(f"Failed to load translation from {p}: {e}")

    st.warning(f"Translation file not found. Searched: {', '.join(str(x) for x in candidates)}")
    return {}

# load translations (default ru)
T = load_translation("ru")

def _(key: str, default: str = "") -> str:
    """Safe lookup for nested keys like 'table_headers.revenue'. Returns default or key when not found."""
    parts = key.split(".")
    cur = T
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default or key
    return cur if isinstance(cur, str) else str(cur)

# ---------------------------
# Helpers
# ---------------------------
def format_rub(value: Any, decimals: int = 0) -> str:
    """Format numeric value as Russian rubles with space thousand separator.
    If value is None -> returns em dash.
    """
    try:
        if value is None:
            return "—"
        v = float(value)
        fmt = f"{v:,.{decimals}f}"
        # replace comma thousands separator with space
        fmt = fmt.replace(",", " ")
        return f"₽{fmt}"
    except Exception:
        return f"₽{value}"

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title=_("app_title", "Scenario — Финансовый симулятор"), layout="wide")
st.title(_("app_title", "Scenario — Финансовый симулятор"))

# ---------------------------
# Sidebar inputs (localized)
# ---------------------------
with st.sidebar:
    st.header(_("inputs_header", "Входные параметры"))

    mrr = st.number_input(_("mrr_label", "MRR (₽ / месяц)"), min_value=0.0, value=5000.0, step=100.0, format="%.2f")
    monthly_growth_pct = st.number_input(_("monthly_growth_label", "Рост в месяц (%)"), min_value=0.0, max_value=1000000.0, value=5.0, step=0.1, format="%.3f")
    churn_pct = st.number_input(_("churn_label", "Отток (%)"), min_value=0.0, max_value=100.0, value=2.0, step=0.1, format="%.3f")
    arpu = st.number_input(_("arpu_label", "ARPU (₽ / пользователь / месяц)"), min_value=0.0, value=50.0, step=1.0, format="%.2f")
    cac = st.number_input(_("cac_label", "CAC (₽)"), min_value=0.0, value=200.0, step=10.0, format="%.2f")
    fixed_costs = st.number_input(_("fixed_costs_label", "Фиксированные расходы (₽ / месяц)"), min_value=0.0, value=3000.0, step=100.0, format="%.2f")
    variable_costs_pct = st.number_input(_("variable_costs_label", "Переменные расходы (%)"), min_value=0.0, max_value=100.0, value=20.0, step=0.1, format="%.3f")
    payment_lag_days = st.number_input(_("payment_lag_label", "Задержка платежа (дней)"), min_value=0, max_value=365, value=30, step=1)
    horizon_months = st.number_input(_("horizon_label", "Горизонт прогноза (месяцев)"), min_value=1, max_value=120, value=12, step=1)

    st.markdown("---")
    allow_send_ai = st.checkbox(_("ui.ai_consent_checkbox_label", "Разрешаю отправку данных в AI (PII не отправляется по умолчанию)"), value=False)

# ---------------------------
# Inputs dict
# ---------------------------
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
# JSON schema (from document D)
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
    "required": ["mrr", "monthly_growth_pct", "churn_pct", "arpu", "cac", "fixed_costs", "variable_costs_pct"],
}

# ---------------------------
# Cached calculations
# ---------------------------
@st.cache_data(ttl=300)
def run_calculations(inp: Dict[str, Any]) -> Dict[str, Any]:
    local = deepcopy(inp)
    validate(instance=local, schema=INPUTS_SCHEMA)

    pnl_df = calculate_pnl(local)
    cf_df = calculate_cashflow(local)

    try:
        unit_metrics = calculate_unit_economics(local)
    except ValueError as e:
        unit_metrics = {
            "ltv": None,
            "ltv_cac": None,
            "break_even_month": None,
            "warnings": [_("warnings.churn_zero", "Отток = 0% — значение нестабильно для LTV")],
        }

    return {"pnl": pnl_df, "cashflow": cf_df, "unit_metrics": unit_metrics}

# ---------------------------
# Run + UI
# ---------------------------
calc_result = None
try:
    with st.spinner(_("ui.calculating_spinner", "Вычисляем...")):
        calc_result = run_calculations(inputs)
except ValidationError as ve:
    st.error(_("errors.invalid_input", "Ошибка: Некорректные входные данные."))
    st.json({"status": "error", "code": "INVALID_INPUT", "message": str(ve)})
except Exception as e:
    st.error(_("errors.internal_error", "Произошла внутренняя ошибка."))
    with st.expander(_("ui.traceback_expander", "Подробности для разработчика")):
        st.exception(e)

if calc_result:
    pnl_df: pd.DataFrame = calc_result["pnl"]
    cf_df: pd.DataFrame = calc_result["cashflow"]
    metrics = calc_result["unit_metrics"]

    # show warnings
    for w in metrics.get("warnings", []):
        st.warning(w)

    left, right = st.columns([1, 2])

    with left:
        st.header(_("unit_econ_header", "Юнит-экономика"))

        ltv = metrics.get("ltv")
        ltv_cac = metrics.get("ltv_cac")
        bem = metrics.get("break_even_month")

        st.metric(_("ltv_label", "LTV"), format_rub(ltv, decimals=0) if ltv is not None else "—")
        st.metric(_("ltv_cac_label", "LTV / CAC"), f"{ltv_cac:.2f}" if (ltv_cac is not None) else "—")
        st.metric(_("bem_metric_label", "Месяц выхода на окупаемость"), str(bem) if bem is not None else "—")

        st.markdown("---")
        if st.button(_("ui.get_ai", "Получить рекомендацию (AI)")):
            if not allow_send_ai:
                st.info(_("ui.ai_consent_required", "Отметьте чекбокс 'Разрешаю отправку данных в AI' в сайдбаре, чтобы использовать AI."))
            else:
                with st.spinner(_("ui.ai_spinner", "Запрошено у AI...")):
                    st.success(_("ui.ai_stub_response", "AI: Сократите CAC и улучшите удержание — это повысит LTV/CAC."))

    with right:
        st.header(_("reports_header", "Отчёты"))

        # glossary
        with st.expander(_("glossary.title", "Термины и обозначения"), expanded=False):
            g = T.get("glossary", {})
            st.write(f"**MRR** — {g.get('mrr', '—')}")
            st.write(f"**ARPU** — {g.get('arpu', '—')}")
            st.write(f"**CAC** — {g.get('cac', '—')}")
            st.write(f"**LTV** — {g.get('ltv', '—')}")
            st.write(f"**Churn** — {g.get('churn', '—')}")
            st.write(f"**COGS** — {g.get('cogs', '—')}")
            st.write(f"**Payment lag** — {g.get('payment_lag', '—')}")
            st.write(f"**Horizon** — {g.get('horizon', '—')}")

        # localized headers map for dataframe display only
        table_headers_translation = T.get("table_headers", {})

        tabs = st.tabs([
            _("pnl_tab", "P&L"),
            _("cashflow_tab", "Денежный поток"),
            _("charts_tab", "Графики"),
            _("export_tab", "Экспорт"),
            _("how_calc_tab", "Как считается"),
        ])

        with tabs[0]:
            st.subheader(_("pnl_subheader", "Отчёт о прибылях и убытках"))
            pnl_df_display = pnl_df.rename(columns=table_headers_translation).rename_axis(_("table_index_month", "Месяц"))
            st.dataframe(pnl_df_display, use_container_width=True)

        with tabs[1]:
            st.subheader(_("cashflow_subheader", "Отчёт о движении денежных средств"))
            cf_df_display = cf_df.rename(columns=table_headers_translation).rename_axis(_("table_index_month", "Месяц"))
            st.dataframe(cf_df_display, use_container_width=True)

        with tabs[2]:
            st.subheader(_("charts_tab", "Графики"))
            
            headers = T.get("table_headers", {})
            index_name = _("table_index_month", "Месяц")

            # --- График выручки ---
            if "revenue" in pnl_df.columns:
                st.subheader(_("charts.revenue", "Выручка по месяцам"))
                revenue_chart_data = (
                    pnl_df[["revenue"]]
                    .rename(columns=headers)
                    .rename_axis(index_name) # <--- ДОБАВЛЕНО ЗДЕСЬ
                )
                st.line_chart(revenue_chart_data, height=250)

            # --- График валовой прибыли ---
            if "gross_profit" in pnl_df.columns:
                st.subheader(_("charts.gross_profit", "Валовая прибыль"))
                gross_profit_chart_data = (
                    pnl_df[["gross_profit"]]
                    .rename(columns=headers)
                    .rename_axis(index_name) # <--- И ЗДЕСЬ
                )
                st.line_chart(gross_profit_chart_data, height=200)

            # --- График поступлений и платежей ---
            if set(["receipts", "payments"]).issubset(cf_df.columns):
                st.subheader(_("charts.receipts_payments", "Поступления и платежи"))
                cashflow_chart_data = (
                    cf_df[["receipts", "payments"]]
                    .rename(columns=headers)
                    .rename_axis(index_name) # <--- И ЗДЕСЬ
                )
                st.line_chart(cashflow_chart_data, height=250)

        with tabs[3]:
            st.subheader(_("export_tab", "Экспорт"))
            st.download_button(_("ui.download_pnl", "Скачать P&L CSV"), pnl_df.to_csv(index=False).encode("utf-8"), "pnl.csv", "text/csv")
            st.download_button(_("ui.download_cf", "Скачать CashFlow CSV"), cf_df.to_csv(index=False).encode("utf-8"), "cashflow.csv", "text/csv")
            st.download_button(_("ui.download_inputs", "Скачать inputs JSON"), json.dumps(inputs, indent=2, ensure_ascii=False).encode("utf-8"), "inputs.json", "application/json")

        with tabs[4]:
            st.subheader(_("how_calc_tab", "Как считается"))
            st.latex(r"\mathrm{LTV} = \frac{ARPU}{Churn\_rate / 100}")
            st.write(_("formulas_description", "- Revenue моделируется как MRR * (1 + monthly_growth)^(t-1)."))
