"""
DOCSTRING: TODO
"""
# %% Imports
import warnings
from typing import List
import pandas as pd
import dateparser
import matplotlib.pyplot as plt
import os
from loguru import logger
from logger_config import logger_config
from dotenv import load_dotenv

load_dotenv()
logger_config(level=os.getenv('LOG_LEVEL'))
pd.options.mode.chained_assignment = None  # default='warn'
plt.style.use('seaborn-notebook')
warnings.filterwarnings(
    "ignore", message="The localize method is no longer necessary, as this time zone supports the fold attribute",)


def convert_int(x):
    return int(x.replace('.', ''))


def to_100(x):  # mirando datos del bc, pib existe entre ~85-120 - igual esto es cm (?)
    x = x.split('.')
    if x[0].startswith('1'):  # es 100+
        if len(x[0]) > 2:
            return float(x[0] + '.' + x[1])
        else:
            x = x[0]+x[1]
            return float(x[0:3] + '.' + x[3:])
    else:
        if len(x[0]) > 2:
            return float(x[0][0:2] + '.' + x[0][-1])
        else:
            x = x[0] + x[1]
            return float(x[0:2] + '.' + x[2:])


# %% Data Ingestion
def ingest_data():
    # Rain data
    rain = pd.read_csv('data_science/data/precipitaciones.csv')  # [mm]
    central_bank = pd.read_csv('data_science/data/banco_central.csv')
    milk_price = pd.read_csv('data_science/data/precio_leche.csv')
    # rain['date'] = pd.to_datetime(rain['date'], format='%Y-%m-%d')
    # rain = rain.sort_values(by='date', ascending=True).reset_index(drop=True)
    # rain[rain.isna().any(axis=1)]  # no tiene nans
    # rain[rain.duplicated(subset='date', keep=False)]  # ni repetidos

    # rain[regions].describe()
    # rain.dtypes

    # Central bank data

    # central_bank['Periodo'] = central_bank['Periodo'].apply(lambda x: x[0:10])
    # central_bank['Periodo'] = pd.to_datetime(
    #     central_bank['Periodo'], format='%Y-%m-%d', errors='coerce')
    # central_bank[central_bank.duplicated(
    #     subset='Periodo', keep=False)]  # repetido se elimina
    # central_bank.drop_duplicates(subset='Periodo', inplace=True)
    # central_bank = central_bank[~central_bank.Periodo.isna()]
    # cols_pib = [x for x in list(central_bank.columns) if 'PIB' in x]
    # cols_pib.extend(['Periodo'])
    # central_bank_pib = central_bank[cols_pib]
    # central_bank_pib = central_bank_pib.dropna(how='any', axis=0)

    # for col in cols_pib:
    #     if col == 'Periodo':
    #         continue
    #     else:
    #         central_bank_pib[col] = central_bank_pib[col].apply(
    #             lambda x: convert_int(x))

    # central_bank_pib.sort_values(by='Periodo', ascending=True)

    return rain, central_bank, milk_price


def validate_data(data, cols: List[str] = None, metadata: bool = False, duplicate_col: str = None):
    if cols:
        for col in cols:
            assert col in data.columns, f'Input data is missing required column {col}.'
    nan_data = {}
    for col in data:
        nan_data[col] = data[col].isnull().sum()
    duplicates = data[data.duplicated(
        subset=duplicate_col, keep=False)] if duplicate_col else None
    return pd.DataFrame(nan_data, index=[0]), duplicates


def preprocessing():
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
        nan_data, dups = validate_data(data[c], required_cols[c])
        nans = nan_data.sum().sum()
        if nans != 0:
            logger.debug(f'{nans} Nan datapoints found.')
            # for col in data[c]:
            #     logger.debug(f'\n{nan_data[col].name}\n{nan_data[col].values[0]}')
        else:
            logger.debug(f'No Nan values found.')
        if dups:
            logger.debug(f'{dups} duplicate rows found.')
        else:
            logger.debug(f'No duplicate rows found.')

    rain['date'] = pd.to_datetime(rain['date'], format='%Y-%m-%d')
    rain = rain.sort_values(by='date', ascending=True).reset_index(drop=True)
    rain = rain.dropna(how='any', axis=0)
    rain = rain.drop_duplicates(subset='date')
    rain['mes'] = rain.date.apply(lambda x: x.month)
    rain['ano'] = rain.date.apply(lambda x: x.year)

    cols_pib = [c for c in central_bank.columns if 'PIB' in c]
    cols_imacec = [x for x in list(central_bank.columns) if 'Imacec' in x]
    data_imc = central_bank[['Periodo',
                            'Indice_de_ventas_comercio_real_no_durables_IVCM']]
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
    # %%
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

    # %%
    data = pd.concat([data['Precio_leche'], data_shift3_mean,
                      data_shift3_std, data_shift1], axis=1)
    data = data.dropna(how='any', axis=0)
    data[['Precio_leche', 'Precio_leche_mes_anterior']]
    return data
