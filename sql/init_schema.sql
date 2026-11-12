-- логические схемы
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS mart;
CREATE SCHEMA IF NOT EXISTS app;

-- таблица для основного датасета (Customer-Churn-Records.csv)
CREATE TABLE IF NOT EXISTS raw.customers_primary (
    row_number INT,
    customer_id BIGINT PRIMARY KEY,
    surname VARCHAR(100),
    credit_score INT NOT NULL,
    geography VARCHAR(50) NOT NULL,
    gender VARCHAR(20) NOT NULL,
    age INT NOT NULL,
    tenure INT NOT NULL,
    balance NUMERIC(15, 2) NOT NULL,
    num_of_products INT NOT NULL,
    has_cr_card INT NOT NULL,
    is_active_member INT NOT NULL,
    estimated_salary NUMERIC(15, 2) NOT NULL,
    exited INT NOT NULL,
    complain INT,
    satisfaction_score INT,
    card_type VARCHAR(30),
    point_earned INT,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- аналитическая витрина для обучения и интерпретации
CREATE OR REPLACE VIEW mart.v_primary_features AS
SELECT
    customer_id,
    credit_score,
    geography,
    gender,
    age,
    tenure,
    balance,
    num_of_products,
    has_cr_card,
    is_active_member,
    estimated_salary,
    satisfaction_score,
    card_type,
    point_earned,
    ROUND((balance / (estimated_salary + 1))::numeric, 6) AS balance_to_salary,
    CASE WHEN is_active_member = 0 AND balance = 0 THEN 1 ELSE 0 END AS inactive_and_zero_balance,
    exited
FROM raw.customers_primary;

-- таблица логирования предсказаний пользователей из веб-интерфейса
CREATE TABLE IF NOT EXISTS app.inference_logs (
    log_id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_used VARCHAR(50) NOT NULL,
    input_features JSONB NOT NULL,
    churn_probability NUMERIC(5, 4) NOT NULL,
    predicted_churn INT NOT NULL,
    threshold_used NUMERIC(4, 3) NOT NULL,
    shap_explanation JSONB,
    llm_explanation TEXT
);