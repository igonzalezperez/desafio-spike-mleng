"""
DOCSTRING: TODO
"""
# %% Imports
import warnings
import json
from typing import List
import pandas as pd
import dateparser
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from loguru import logger
from src.logger_config import logger_config
from src.utils.utils import convert_int, to_100

# %% Config
load_dotenv()
logger_config(level=os.getenv('LOG_LEVEL'))
pd.options.mode.chained_assignment = None  # default='warn'
plt.style.use('seaborn-notebook')
warnings.filterwarnings(
    "ignore", message="The localize method is no longer necessary, as this time zone supports the fold attribute",)


# %% Data Ingestion
def ingest_data():
    rain = pd.read_csv('data/precipitaciones.csv')
    central_bank = pd.read_csv('data/banco_central.csv')
    milk_price = pd.read_csv('data/precio_leche.csv')
    return rain, central_bank, milk_price


# %% Data validation
def validate_data(data, cols: List[str] = None, metadata: bool = False, duplicate_col: str = None):
    if cols:
        for col in cols:
            assert col in data.columns, f'Input data is missing required column {col}.'
    nan_data = {}
    for col in data:
        nan_data[col] = data[col].isnull().sum()
    duplicates = data[data.duplicated(
        subset=duplicate_col, keep=False)] if duplicate_col else data[data.duplicated(keep=False)]
    return pd.DataFrame(nan_data, index=[0]), duplicates


# %% Data cleaning and formatting
def prepare_data(rain, central_bank, milk_price):
    rain['date'] = pd.to_datetime(rain['date'], format='%Y-%m-%d')
    rain = rain.sort_values(by='date', ascending=True).reset_index(drop=True)
    rain = rain.dropna(how='any', axis=0)
    rain = rain.drop_duplicates(subset='date')
    rain['mes'] = rain.date.apply(lambda x: x.month)
    rain['ano'] = rain.date.apply(lambda x: x.year)
    breakpoint
    cols_pib = [c for c in central_bank.columns if 'PIB' in c]
    cols_imacec = [x for x in list(central_bank.columns) if 'Imacec' in x]
    central_bank['Periodo'] = central_bank['Periodo'].apply(lambda x: x[0:10])
    central_bank['Periodo'] = pd.to_datetime(
        central_bank['Periodo'], format='%Y-%m-%d', errors='coerce')

    central_bank = central_bank.drop_duplicates(subset='Periodo')
    central_bank = central_bank[~central_bank['Periodo'].isna()]
    central_bank = central_bank[['Periodo', *cols_pib, *cols_imacec,
                                'Indice_de_ventas_comercio_real_no_durables_IVCM']]
    central_bank = central_bank.dropna(how='any', axis=0)

    for col in central_bank:
        if col in cols_pib:
            central_bank[col] = central_bank[col].apply(
                lambda x: convert_int(x))
        elif col in cols_imacec:
            central_bank[col] = central_bank[col].apply(lambda x: to_100(x))
            assert central_bank[col].max() > 100
            assert central_bank[col].min() > 30
    central_bank['num'] = central_bank['Indice_de_ventas_comercio_real_no_durables_IVCM'].apply(
        lambda x: to_100(x))
    central_bank = central_bank.sort_values(by='Periodo', ascending=True)
    central_bank['mes'] = central_bank['Periodo'].apply(lambda x: x.month)
    central_bank['ano'] = central_bank['Periodo'].apply(lambda x: x.year)

    milk_price.rename(columns={'Anio': 'ano', 'Mes': 'mes_pal'}, inplace=True)
    milk_price['mes'] = milk_price['mes_pal'].apply(
        lambda x: dateparser.parse(x))
    milk_price['mes'] = milk_price['mes'].apply(lambda x: x.month)
    milk_price['mes-ano'] = milk_price.apply(
        lambda x: f'{x.mes}-{x.ano}', axis=1)
    data = rain.merge(central_bank, how='inner', on=['mes', 'ano'])
    data = data.merge(milk_price, how='inner', on=['mes', 'ano'])
    data.drop(['date', 'Periodo', 'Indice_de_ventas_comercio_real_no_durables_IVCM',
               'mes-ano', 'mes_pal'], axis=1, inplace=True)
    year_col = data.pop('ano')
    month_col = data.pop('mes')
    data.insert(0, 'mes', month_col)
    data.insert(0, 'ano', year_col)
    return data


# %% Custom data transformation (moving averages)
def moving_avg_transform(data):
    cc_cols = [x for x in data.columns if x not in ['ano', 'mes']]

    # %%
    data_shift3_mean = data[cc_cols].rolling(
        window=3, min_periods=1).mean().shift(1)

    data_shift3_mean.columns = [
        x + '_shift3_mean' for x in data_shift3_mean.columns]

    data_shift3_std = data[cc_cols].rolling(
        window=3, min_periods=1).std().shift(1)

    data_shift3_std.columns = [
        x + '_shift3_std' for x in data_shift3_std.columns]

    data_shift1 = data[cc_cols].shift(1)
    data_shift1.columns = [x + '_mes_anterior' for x in data_shift1.columns]
    data = pd.concat([data['Precio_leche'], data_shift3_mean,
                      data_shift3_std, data_shift1], axis=1)
    data = data.dropna(how='any', axis=0)
    return data


def preprocess():
    logger.debug('Data ingestion - Rain, Central bank and milk price.')
    rain, central_bank, milk_price = ingest_data()
    data = {'rain': rain, 'central_bank': central_bank, 'milk_price': milk_price}
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
