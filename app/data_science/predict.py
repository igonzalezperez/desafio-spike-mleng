from typing import List

from joblib import load
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta as rdelta
from sklearn.metrics import r2_score
from database.database import get_db_data
from data_science.utils.utils import datetime_to_unix
from data_science.preprocessing import DataPreparation
from loguru import logger

import config.config as cfg


def make_predictions(pred_dates: List[pd.Timestamp]) -> pd.DataFrame:
    """
    Predict the price of milk using trained preprocess and model pipelines for a given list of dates.

    Args:
        pred_dates (pd.Timestamp): List of dates to predict.

    Returns:
        pd.DataFrame: Predictions for each required date.
    """
    if pred_dates:
        pred_dates.sort()
        prev_dates = []
        for d in pred_dates:
            pvd = [datetime_to_unix(d - rdelta(months=i)) for i in range(1, 4)]
            [prev_dates.append(p) for p in pvd if p not in prev_dates]
        prev_dates = list(set(prev_dates))
        prev_dates.sort()
        pred_dates = [datetime_to_unix(prd) for prd in pred_dates]

        x, y = get_db_data()

        y_ = y[y['timestamp'].isin(pred_dates)]

        x = x[x['timestamp'].isin(prev_dates)]
        y = y[y['timestamp'].isin(prev_dates)]

        idx = [pd.to_datetime(prd, unit='s') for prd in pred_dates]
        idx = [f'{i.year}-{i.month:02d}' for i in idx]
        if len(x) < len(pred_dates) + 2:
            missing_rows = [pd.to_datetime(i, unit='s').strftime(
                '%Y-%m') for i in prev_dates if i not in x['timestamp'].values]
            logger.error(f'Missing feature data {", ".join(missing_rows)}.')
            return (None, f'Faltan datos (features) de algunos meses para predecir: {", ".join(missing_rows)}.')
        if len(x) < len(pred_dates) + 2:
            missing_rows = [
                i for i in prev_dates if i not in y['timestamp'].values]
            logger.error(
                f'Missing target data in months: {", ".join(missing_rows)}.')
            return (None, f'Faltan datos (target) de algunos meses para predecir: {", ".join(missing_rows)}.')
        feature_pipeline = load(cfg.FEATURE_PIPELINE_PATH)
        model = load(cfg.MODEL_PATH)
        x, y = DataPreparation().transform(x, y, mode='predict')
        x = feature_pipeline.transform(x)
        preds = model.predict(x)
        output = pd.DataFrame()
        output['Fecha'] = idx
        output['Precio pred'] = preds
        idx = [pd.to_datetime(prd, unit='s') for prd in y_['timestamp']]
        idx = [f'{i.year}-{i.month:02d}' for i in idx]
        y_['timestamp'] = idx
        y_ = y_.rename(
            {'timestamp': 'Fecha', 'Precio_leche': 'Precio real'}, axis=1)
        output = output.merge(y_, how='outer', on='Fecha')
        logger.info(f'Data predicted successfully.')
        logger.info(f"""Preds:
    {output}""")

    else:
        x, y = get_db_data()
        x, y = DataPreparation().transform(x, y)

        feature_pipeline = load(cfg.FEATURE_PIPELINE_PATH)
        model = load(cfg.MODEL_PATH)
        x = feature_pipeline.transform(x)

        preds = model.predict(x)

        rmse_all = np.sqrt(((y - preds) ** 2).mean())
        r2_all = r2_score(y, preds)

        logger.info('Results:')
        logger.info(f'RMSE: {rmse_all:.2f} (all data)')
        logger.info(f'R2: {r2_all:.2f} (all data)')

        output = pd.DataFrame()
    return output, None
