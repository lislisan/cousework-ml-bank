"""Reproducible ingestion, EDA and baseline training for bank churn."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
CSV_COLUMNS = {
    'RowNumber': 'row_number', 'CustomerId': 'customer_id', 'Surname': 'surname',
    'CreditScore': 'credit_score', 'Geography': 'geography', 'Gender': 'gender',
    'Age': 'age', 'Tenure': 'tenure', 'Balance': 'balance',
    'NumOfProducts': 'num_of_products', 'HasCrCard': 'has_cr_card',
    'IsActiveMember': 'is_active_member', 'EstimatedSalary': 'estimated_salary',
    'Exited': 'exited', 'Complain': 'complain',
    'Satisfaction Score': 'satisfaction_score', 'Card Type': 'card_type',
    'Point Earned': 'point_earned',
}
OPTIONAL = {'RowNumber', 'Surname', 'Complain', 'Satisfaction Score', 'Card Type', 'Point Earned'}
TARGET = 'exited'
EXCLUDED = {'customer_id', TARGET, 'complain'}


def settings():
    load_dotenv(ROOT / '.env')
    with (ROOT / 'config/config.yaml').open(encoding='utf-8') as file:
        return yaml.safe_load(file)


def engine():
    return create_engine(URL.create(
        'postgresql+psycopg2',
        username=os.getenv('POSTGRES_USER', 'churn_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'churn_password'),
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', '5432')),
        database=os.getenv('POSTGRES_DB', 'churn_db'),
    ), pool_pre_ping=True)


def ingest(db, csv_path: Path):
    if not csv_path.is_file():
        raise FileNotFoundError(f'CSV not found: {csv_path}. Put the original dataset there or pass --csv.')
    frame = pd.read_csv(csv_path)
    missing = set(CSV_COLUMNS) - OPTIONAL - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required CSV columns: {sorted(missing)}')
    frame = frame.rename(columns=CSV_COLUMNS)
    for column in CSV_COLUMNS.values():
        if column not in frame:
            frame[column] = None
    frame = frame[list(CSV_COLUMNS.values())]
    if frame.customer_id.isna().any() or frame.customer_id.duplicated().any():
        raise ValueError('customer_id must be present and unique')
    if frame.exited.isna().any() or not set(frame.exited.unique()).issubset({0, 1}):
        raise ValueError('exited must contain only 0 and 1')
    required = [CSV_COLUMNS[c] for c in set(CSV_COLUMNS) - OPTIONAL]
    if frame[required].isna().any().any():
        raise ValueError('Required source fields contain missing values')
    # Replace the snapshot atomically. Re-running ingestion cannot duplicate rows.
    with db.begin() as connection:
        connection.execute(text('DELETE FROM raw.customers_primary'))
        frame.to_sql('customers_primary', connection, schema='raw', if_exists='append', index=False,
                     chunksize=1000, method='multi')
    print(f'Loaded {len(frame)} rows from {csv_path}')


def load_mart(db, view: str):
    if view != 'mart.v_primary_features':
        raise ValueError('Only the configured primary feature view is supported')
    frame = pd.read_sql_query(text('SELECT * FROM mart.v_primary_features ORDER BY customer_id'), db)
    if frame.empty:
        raise ValueError('The mart is empty. Run ingest first.')
    if frame[TARGET].isna().any() or frame[TARGET].nunique() != 2:
        raise ValueError('Target must have two non-null classes')
    return frame


def eda(frame: pd.DataFrame, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    features = frame.drop(columns=['customer_id'])
    summary = {
        'rows': len(frame), 'columns': list(frame.columns),
        'duplicate_customer_ids': int(frame.customer_id.duplicated().sum()),
        'missing_values': {k: int(v) for k, v in features.isna().sum().items()},
        'target_counts': {str(k): int(v) for k, v in frame[TARGET].value_counts().items()},
        'numeric_description': json.loads(features.describe().to_json()),
        'churn_by_geography': frame.groupby('geography')[TARGET].agg(['count', 'mean']).round(4).to_dict('index'),
        'churn_by_products': {str(k): v for k, v in frame.groupby('num_of_products')[TARGET].agg(['count', 'mean']).round(4).to_dict('index').items()},
        'churn_by_activity': {str(k): v for k, v in frame.groupby('is_active_member')[TARGET].agg(['count', 'mean']).round(4).to_dict('index').items()},
    }
    (output / 'eda.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'EDA report: {output / "eda.json"}')


def train(frame: pd.DataFrame, cfg: dict, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    x = frame.drop(columns=list(EXCLUDED), errors='ignore')
    y = frame[TARGET].astype(int)
    x = x.dropna(axis=1, how='all')
    numeric = list(x.select_dtypes(include=np.number).columns)
    categorical = [c for c in x.columns if c not in numeric]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=cfg['test_size'], random_state=cfg['random_state'], stratify=y)
    ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    seed = cfg['random_state']
    models = {
        'logistic_regression': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=seed),
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10,
                                                 min_samples_leaf=5, class_weight='balanced', random_state=seed, n_jobs=-1),
        'xgboost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                  scale_pos_weight=ratio, eval_metric='logloss', random_state=seed, n_jobs=1),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                         learning_rate=0.1, random_state=seed),
        'svm': SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=seed),
    }
    rows = []
    best_score, best_name, best_pipeline = -1.0, None, None
    for name, estimator in models.items():
        preprocessor = ColumnTransformer([
            ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                                  ('scale', StandardScaler())]), numeric),
            ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                      ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical),
        ])
        pipeline = Pipeline([('preprocess', preprocessor), ('model', estimator)])
        start = time.monotonic()
        pipeline.fit(x_train, y_train)
        duration = time.monotonic() - start
        prediction = pipeline.predict(x_test)
        probability = pipeline.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, probability)
        tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
        rows.append({'model': name, 'train_seconds': round(duration, 3),
                     'accuracy': accuracy_score(y_test, prediction),
                     'precision': precision_score(y_test, prediction, zero_division=0),
                     'recall': recall_score(y_test, prediction, zero_division=0),
                     'f1': f1_score(y_test, prediction, zero_division=0),
                     'roc_auc': auc, 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)})
        if auc > best_score:
            best_score, best_name, best_pipeline = auc, name, pipeline
    results = pd.DataFrame(rows).sort_values('roc_auc', ascending=False)
    results.to_csv(output / 'model_metrics.csv', index=False)
    joblib.dump({'pipeline': best_pipeline, 'model_name': best_name,
                 'features': list(x.columns), 'decision_threshold': 0.5},
                output / cfg['best_model_filename'])
    metadata = {'source': 'mart.v_primary_features', 'excluded_columns': sorted(EXCLUDED),
                'train_rows': len(x_train), 'test_rows': len(x_test), 'random_state': seed,
                'test_size': cfg['test_size'], 'selection_metric': 'roc_auc',
                'best_model': best_name, 'threshold': 0.5,
                'note': 'Test metrics are descriptive; selecting a model on this split makes its reported score optimistic.'}
    (output / 'run_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(results.round(4).to_string(index=False))
    print(f'Model saved: {output / cfg["best_model_filename"]}')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['ingest', 'eda', 'train', 'all'])
    parser.add_argument('--csv', type=Path, help='Path to Customer-Churn-Records.csv')
    args = parser.parse_args()
    cfg = settings()
    db = engine()
    csv = args.csv or ROOT / cfg['data']['raw_dir'] / cfg['data']['raw_filename']
    if args.command in ('ingest', 'all'):
        ingest(db, csv)
    if args.command in ('eda', 'train', 'all'):
        frame = load_mart(db, cfg['data']['source_view'])
        if args.command in ('eda', 'all'):
            eda(frame, ROOT / 'reports')
        if args.command in ('train', 'all'):
            train(frame, cfg['model'], ROOT / cfg['model']['artifacts_dir'])
    db.dispose()


if __name__ == '__main__':
    main()
