# Scenario-Core: Финансовый симулятор

Основной движок для финансового моделирования и симуляций, написанный на Python с использованием Streamlit и Pandas.

## ✨ Ключевые возможности

-   Интерактивный UI для ввода бизнес-метрик (MRR, ARPU, CAC, Churn и др.).
-   Автоматический расчет P&L, Cash Flow и ключевых метрик юнит-экономики (LTV, LTV/CAC).
-   Визуализация данных с помощью интерактивных графиков.
-   Экспорт полного отчета (P&L, Cash Flow, параметры) в отформатированный многостраничный Excel-файл.
-   Полная локализация интерфейса (русский язык).

## 🚀 Локальный запуск

1.  **Клонируйте репозиторий:**
    ```bash
    git clone https://github.com/IDSidorov-data/scenario-core.git
    cd scenario-core
    ```

2.  **Создайте и активируйте виртуальное окружение:**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\Activate.ps1
    # macOS / Linux
    source venv/bin/activate
    ```

3.  **Установите зависимости:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Запустите приложение:**
    ```bash
    streamlit run app.py
    ```

## 🧪 Тестирование

Для запуска тестов установите зависимости для разработки и выполните команду:

```bash
pip install -r requirements-dev.txt
pytest
