import pandas as pd
from dateutil.relativedelta import relativedelta as rdelta
from src.utils.utils import datetime_to_unix
from src.preprocessing import DataPreparation
import sqlalchemy as db
from sqlalchemy import select
from loguru import logger


def make_prediction(date: str):
    pred_date = pd.to_datetime(date, format='%Y-%m')
    prev_dates = [datetime_to_unix(pred_date - rdelta(months=i))
                  for i in range(1, 4)]
    pred_date = datetime_to_unix(pred_date)

    engine = db.create_engine('sqlite:///data/database.db', echo=True)
    conn = engine.connect()
    x = pd.read_sql_table('features', conn)
    y = pd.read_sql_table('target', conn)
    conn.close()
    engine.dispose()

    x = x[x['timestamp'].isin(prev_dates)]
    y = y[y['timestamp'].isin(prev_dates)]
    if len(x) < 3:
        missing_rows = [pd.to_datetime(i, unit='s').strftime(
            '%Y-%m') for i in prev_dates if i not in x['timestamp'].values]
        logger.debug(f'Missing feature data {missing_rows}.')
        return
    if len(y) < 3:
        missing_rows = [
            i for i in prev_dates if i not in y['timestamp'].values]
        logger.debug(f'Missing target data {missing_rows}.')
        return
    x, y = DataPreparation().transform(x, y, mode='predict')
    breakpoint()
    return x, y
