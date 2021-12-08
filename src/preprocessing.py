"""
DOCSTRING: TODO
"""
# %% Imports
import os
import warnings
import json
from typing import List

import dateparser
from dotenv import load_dotenv
from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
import sqlalchemy as db
from dateutil.relativedelta import relativedelta as rdelta
from src.utils.logger_config import logger_config
from src.utils.utils import convert_int, to_100, datetime_to_unix

# %% Config
load_dotenv()
logger_config(level=os.getenv('LOG_LEVEL'))
pd.options.mode.chained_assignment = None  # default='warn'
plt.style.use('seaborn-notebook')
warnings.filterwarnings(
    "ignore", message="The localize method is no longer necessary, as this time zone supports the fold attribute",)


# %% Classes and functions
class DataPreparation(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None, mode='train'):
        data = x.merge(y, how='inner', on='timestamp')
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
        data.set_index(['timestamp'], inplace=True)
        x, y = moving_avg_transform(data, mode)
        return x, y


def create_db(mode='fail'):
    engine = db.create_engine('sqlite:///data/database.db', echo=True)
    conn = engine.connect()
    data = ingest_data()
    x, y = prepare_data(data)
    try:
        x.to_sql(name='features', con=conn,
                 if_exists=mode, index=False)
        y.to_sql(name='target', con=conn,
                 if_exists=mode, index=False)
    except ValueError:
        logger.info(
            "Tables 'features' and 'target' already exist. Set 'mode='replace'' to overwrite.")
    conn.close()
    engine.dispose()


def get_data():
    engine = db.create_engine('sqlite:///data/database.db', echo=True)
    conn = engine.connect()
    x = pd.read_sql_table('features', conn)
    y = pd.read_sql_table('target', conn)
    conn.close()
    engine.dispose()
    return x, y


def ingest_data():
    rain = pd.read_csv('data/precipitaciones.csv')
    central_bank = pd.read_csv('data/banco_central.csv')
    milk_price = pd.read_csv('data/precio_leche.csv')
    return {'rain': rain, 'central_bank': central_bank, 'milk_price': milk_price}


def validate_data(data, cols: List[str] = None, metadata: bool = False, duplicate_col: str = None):
    if cols:
        for col in cols:
            assert col in data.columns, f'Input data is missing required column {col}.'
    nan_data = {}
    for col in data:
        nan_data[col] = data[col].isnull().sum()
    duplicates = data[data.duplicated(
        subset=duplicate_col, keep=False)] if duplicate_col else data[data.duplicated(keep=False)]
    return pd.dataFrame(nan_data, index=[0]), duplicates


def prepare_data(data):
    rain = data['rain']
    central_bank = data['central_bank']
    milk_price = data['milk_price']

    rain['date'] = pd.to_datetime(rain['date'], format='%Y-%m-%d')
    rain = rain.dropna(how='any', axis=0)
    rain = rain.drop_duplicates(subset='date')
    rain = rain.rename({'date': 'timestamp'}, axis=1)

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
    features = rain.merge(central_bank, how='inner', on=['timestamp'])
    features = features.sort_values(by='timestamp')
    milk_price['timestamp'] = milk_price.agg(
        lambda x: f"{x['Anio']}, {x['Mes']} 1", axis=1)
    milk_price['timestamp'] = milk_price['timestamp'].apply(
        lambda x: dateparser.parse(x, languages=['es']))
    target = milk_price.sort_values(by='timestamp')
    target.drop(['Anio', 'Mes'], axis=1, inplace=True)
    for col in features:
        if col in cols_pib:
            features[col] = features[col].apply(
                lambda x: convert_int(x))
        elif col in cols_imacec:
            features[col] = features[col].apply(lambda x: to_100(x))
            assert features[col].max() > 100
            assert features[col].min() > 30
    features['IVCM_num'] = features['Indice_de_ventas_comercio_real_no_durables_IVCM'].apply(
        lambda x: to_100(x))
    features.drop(
        ['Indice_de_ventas_comercio_real_no_durables_IVCM'], axis=1, inplace=True)
    features['timestamp'] = datetime_to_unix(features['timestamp'])
    target['timestamp'] = datetime_to_unix(target['timestamp'])

    time_col = features.pop('timestamp')
    features.insert(0, 'timestamp', time_col)

    time_col = target.pop('timestamp')
    target.insert(0, 'timestamp', time_col)
    return features.reset_index(drop=True), target.reset_index(drop=True)


# %% Custom data transformation (moving averages)
def moving_avg_transform(data, mode='train'):
    if mode == 'train':
        offset = 1
        min_p = 1
        index = data.index[2:]
    elif mode == 'predict':
        offset = 0
        min_p = 3
        index = pd.DatetimeIndex([data.index[-1] + rdelta(months=1)])
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


def preprocess():
    logger.debug('data ingestion - Rain, Central bank and milk price.')
    rain, central_bank, milk_price = ingest_data()
    data = {'rain': rain, 'central_bank': central_bank,
            'milk_price': milk_price}
    regions = ['Coquimbo', 'Valparaiso', 'Metropolitana_de_Santiago',
               'Libertador_Gral__Bernardo_O_Higgins', 'Maule', 'Biobio',
               'La_Araucania', 'Los_Rios']
    required_cols = {'rain': regions, 'central_bank': None, 'milk_price': None}
    duplicate_cols = {'rain': 'date',
                      'central_bank': 'Periodo', 'milk_price': None}

    for d, c in zip(data, required_cols):
        logger.debug(f'{d} data validation.')
        nan_data, dups = validate_data(
            data[c], required_cols[c], False, duplicate_col=duplicate_cols[d])
        nans = nan_data.sum().sum()
        if nans != 0:
            logger.debug(f'{nans} Nan datapoints found.')
        else:
            logger.debug(f'No Nan values found.')
        if not dups.empty:
            logger.debug(f'{len(dups)} duplicate rows found.')
        else:
            logger.debug(f'No duplicate rows found.')
    data = prepare_data(rain, central_bank, milk_price)
    save_data_as_request(data)
    data = moving_avg_transform(data)
    return data


def save_data_as_request(data):
    for i, j in enumerate(range(0, len(data) - 3)):
        data_dict = data.iloc[j: j + 3, :].to_dict('list')
        with open(f'.requests/sample_request_{i}.json', 'w') as handle:
            json.dump(data_dict, handle)
