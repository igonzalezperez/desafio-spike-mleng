"""
Ingest, clean and transform data for training/predicting
"""
# %% Imports
from typing import Tuple, Dict

import dateparser
import pandas as pd
import sqlalchemy as db
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from dateutil.relativedelta import relativedelta as rdelta

from data_science.utils.utils import convert_int, to_100, datetime_to_unix


# %% Classes and functions
class DataPreparation(BaseEstimator, TransformerMixin):
    """
    Prepare input data for training/predicting. Input data comes from database, so it is clean already.
    """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x: pd.DataFrame, y: pd.DataFrame = None, mode: str = 'train') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform data to create model input. Merge features and target, convert unix timestamp to datetime and calculate moving average features.

        Args:
            x (pd.DataFrame): Feature matrix
            y (pd.DataFrame, optional): Target column. Defaults to None.
            mode (str, optional): For which purpose is the transformer, either 'train' or 'predict'. Defaults to 'train'.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Transformed features and target. Target is not used when predicting.
        """
        assert mode in ['train', 'predict'], logger.error(
            "'mode' parameter must be either 'train' or 'predict'.")
        data = x.merge(y, how='inner', on='timestamp')
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
        data.set_index(['timestamp'], inplace=True)
        x, y = moving_avg_transform(data, mode)
        return x, y


def ingest_data() -> Dict[str, pd.DataFrame]:
    """
    Load data from local .csv files.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with one key-value pair per file storing data as DataFrame.
    """
    rain = pd.read_csv('data/precipitaciones.csv')
    central_bank = pd.read_csv('data/banco_central.csv')
    milk_price = pd.read_csv('data/precio_leche.csv')
    return {'rain': rain, 'central_bank': central_bank, 'milk_price': milk_price}


def prepare_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Data preparation for input data (from .csv) to be stored in the DB. Select necessary columns, transform date columns to unix timestamps and numeric columns to float. Drop duplicates and Nans. This function receives the 
    Args:
        data (pd.DataFrame): Dictionary with DataFrames. Output from ingest_data().

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Clean features and target DataFrames with necessaary columns for training.
    """
    rain = data['rain']
    central_bank = data['central_bank']
    milk_price = data['milk_price']
    # Clean rain data - create datetime, drop duplicates and Nans.
    rain['date'] = pd.to_datetime(rain['date'], format='%Y-%m-%d')
    rain = rain.dropna(how='any', axis=0)
    rain = rain.drop_duplicates(subset='date')
    rain = rain.rename({'date': 'timestamp'}, axis=1)

    # Clean central bank data - select needed columns, drop duplicates and nans, create datetime.
    cols_pib = [c for c in central_bank.columns if 'PIB' in c]
    cols_imacec = [x for x in list(central_bank.columns) if 'Imacec' in x]
    central_bank['Periodo'] = central_bank['Periodo'].apply(lambda x: x[0:10])
    central_bank['Periodo'] = pd.to_datetime(
        central_bank['Periodo'], format='%Y-%m-%d', errors='coerce')
    central_bank = central_bank.drop_duplicates(subset='Periodo')
    central_bank = central_bank[~central_bank['Periodo'].isna()]
    central_bank = central_bank[[
        'Periodo', *cols_pib, *cols_imacec, 'Indice_de_ventas_comercio_real_no_durables_IVCM']]
    central_bank = central_bank.rename({'Periodo': 'timestamp'}, axis=1)
    central_bank = central_bank.dropna(how='any', axis=0)

    # Merge (inner join by timestamp) rain and central bank data to features DataFrame - convert str values to float,
    # convert datetime to unix timsetamp.
    features = rain.merge(central_bank, how='inner', on=['timestamp'])
    features = features.sort_values(by='timestamp')

    for col in features:
        if col in cols_pib:
            features[col] = features[col].apply(
                lambda x: convert_int(x))
        elif col in cols_imacec:
            features[col] = features[col].apply(lambda x: to_100(x))
            # assert features[col].max() > 100
            # assert features[col].min() > 30
    features['IVCM_num'] = features['Indice_de_ventas_comercio_real_no_durables_IVCM'].apply(
        lambda x: to_100(x))
    features.drop(
        ['Indice_de_ventas_comercio_real_no_durables_IVCM'], axis=1, inplace=True)
    features['timestamp'] = datetime_to_unix(features['timestamp'])
    time_col = features.pop('timestamp')
    features.insert(0, 'timestamp', time_col)

    # Clean milk price data - Create datetime, convert to unix timestamp.
    milk_price['timestamp'] = milk_price.agg(
        lambda x: f"{x['Anio']}, {x['Mes']} 1", axis=1)
    milk_price['timestamp'] = milk_price['timestamp'].apply(
        lambda x: dateparser.parse(x, languages=['es']))
    target = milk_price.sort_values(by='timestamp')
    target.drop(['Anio', 'Mes'], axis=1, inplace=True)
    target['timestamp'] = datetime_to_unix(target['timestamp'])
    time_col = target.pop('timestamp')
    target.insert(0, 'timestamp', time_col)

    return features.reset_index(drop=True), target.reset_index(drop=True)


def moving_avg_transform(data: pd.DataFrame, mode: str = 'train') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate moving average features for training. To predict y[t + 1] it's necessary to have:
    x[t]
    mean(x[t, t-1, t-2])
    std(x[t, t-1, t-2])

    y[t]
    mean(y[t, t-1, t-2])
    std(y[t, t-1, t-2])

    Args:
        data (pd.DataFrame): Dataframe with features and target data.
        mode (str, optional): Whether the processing is for training or predicting. Defaults to 'train'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Transformed feature data with extra columns for the moving averages.
    """
    assert mode in ['train', 'predict'], logger.error(
        "'mode' parameter must be either 'train' or 'predict'.")
    if mode == 'train':
        offset = 1
        min_p = 1
        index = data.index[2:]
    elif mode == 'predict':
        offset = 0
        min_p = 3
        index = pd.DatetimeIndex([idx + rdelta(months=1)
                                 for idx in data.index[2:]])
    cc_cols = [x for x in data.columns if x != 'timestamp']

    data_shift3_mean = data[cc_cols].rolling(
        window=3, min_periods=min_p).mean().shift(offset)

    data_shift3_mean.columns = [
        x + '_shift3_mean' for x in data_shift3_mean.columns]

    data_shift3_std = data[cc_cols].rolling(
        window=3, min_periods=min_p).std().shift(offset)

    data_shift3_std.columns = [
        x + '_shift3_std' for x in data_shift3_std.columns]

    data_shift1 = data[cc_cols].shift(offset)
    data_shift1.columns = [
        x + '_mes_anterior' for x in data_shift1.columns]
    data = pd.concat([data['Precio_leche'], data_shift3_mean,
                      data_shift3_std, data_shift1], axis=1)
    data = data.dropna(how='any', axis=0)
    data.index = index
    x = data[[x for x in data.columns if x != 'Precio_leche']]
    y = data['Precio_leche']
    return x, y

# %%
