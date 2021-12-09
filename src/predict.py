from joblib import load
import pandas as pd
from dateutil.relativedelta import relativedelta as rdelta
from src.utils.utils import datetime_to_unix
from src.preprocessing import DataPreparation
import sqlalchemy as db
from loguru import logger

feature_pipeline = load('models/pipelines/feature_pipeline.pkl')
target_pipeline = load('models/pipelines/target_pipeline.pkl')
model = load('models/ridge_model.pkl')


def make_predictions(pred_dates: str):
    pred_dates.sort()
    prev_dates = []
    for d in pred_dates:
        pvd = [datetime_to_unix(d - rdelta(months=i)) for i in range(1, 4)]
        [prev_dates.append(p) for p in pvd if p not in prev_dates]
    prev_dates = list(set(prev_dates))
    prev_dates.sort()
    pred_dates = [datetime_to_unix(prd) for prd in pred_dates]

    engine = db.create_engine('sqlite:///data/database.db', echo=True)
    conn = engine.connect()
    x = pd.read_sql_table('features', conn)
    y = pd.read_sql_table('target', conn)
    conn.close()
    engine.dispose()

    x = x[x['timestamp'].isin(prev_dates)]
    y = y[y['timestamp'].isin(prev_dates)]

    if len(x) < len(pred_dates) + 2:
        missing_rows = [pd.to_datetime(i, unit='s').strftime(
            '%Y-%m') for i in prev_dates if i not in x['timestamp'].values]
        logger.debug(f'Missing feature data {missing_rows}.')
        return (None, f'Missing feature data in months: {missing_rows}.')
    if len(x) < len(pred_dates) + 2:
        missing_rows = [
            i for i in prev_dates if i not in y['timestamp'].values]
        logger.debug(f'Missing target data in months: {missing_rows}.')
        return (None, f'Missing target data {missing_rows}.')
    x, _ = DataPreparation().transform(x, y, mode='predict')
    x = feature_pipeline.transform(x)
    preds = model.predict(x)
    preds = target_pipeline.inverse_transform(preds)
    idx = [pd.to_datetime(prd, unit='s') for prd in pred_dates]
    idx = [f'{i.year}-{i.month:02d}' for i in idx]
    output = pd.DataFrame()
    output['Fecha'] = idx
    output['Precio'] = preds

    return output, None
