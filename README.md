# Прогнозирование оттока клиентов банка

На данный момент реализованно: PostgreSQL, Docker, схема данных, загрузка CSV, аналитическая витрина, EDA, подготовка признаков и сравнение пяти базовых моделей. SHAP, LIME, дополнительные датасеты и LLM-анализ здесь пока не реализованы

## Запуск

```bash
docker compose up -d db
docker compose --profile pipeline run --rm app python -m src.pipeline all
```

Отдельные этапы:

```bash
docker compose --profile pipeline run --rm app python -m src.pipeline ingest
docker compose --profile pipeline run --rm app python -m src.pipeline eda
docker compose --profile pipeline run --rm app python -m src.pipeline train
```