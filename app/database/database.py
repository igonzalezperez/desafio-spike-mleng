import os
from typing import Tuple, Dict
import json

import pandas as pd
from loguru import logger
import sqlalchemy as db

from data_science.preprocessing import ingest_data
from data_science.preprocessing import prepare_data

import config.config as cfg


def create_db(mode: str = 'fail', x_in: pd.DataFrame = None, y_in: pd.DataFrame = None) -> None:
    """
    Create an sqlite databse using the original .csv data.

    Args:
        mode (str, optional): What to do if DB already exists, it can be set to 'replace' to reset DB or to 'append' to add more rows. Defaults to 'fail'.
        x_in (pd.DataFrame, optional): Feature data to be inserted if mode is set to 'append'. Defaults to None.
        y_in (pd.DataFrame, optional): Target data to be inserted if mode is set to 'append'. Defaults to None.
    """
    engine = db.create_engine(f'sqlite:///{cfg.DB_PATH}', echo=False)
    conn = engine.connect()
    if mode == 'append':
        if x_in is not None:
            x_in.to_sql(name='features', con=conn,
                        if_exists=mode, index=False)
        if y_in is not None:
            y_in.to_sql(name='target', con=conn,
                        if_exists=mode, index=False)
        return
    data = ingest_data()
    x, y = prepare_data(data)
    try:
        x.to_sql(name='features', con=conn,
                 if_exists=mode, index=False)
        y.to_sql(name='target', con=conn,
                 if_exists=mode, index=False)
    except ValueError:
        logger.warning(
            """Tables 'features' and 'target' already exist. Set 'mode="replace"' to overwrite.""")
    conn.close()
    engine.dispose()
    return


def get_db_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch data from the database's tables (features and target).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Data for features and target from de database as DataFrames.
    """
    engine = db.create_engine(f'sqlite:///{cfg.DB_PATH}', echo=False)
    conn = engine.connect()
    x = pd.read_sql_table('features', conn)
    y = pd.read_sql_table('target', conn)
    conn.close()
    engine.dispose()
    return x, y


def insert_rows(data_dict: Dict[str, pd.DataFrame]) -> str:
    """
    Insert new rows to database if they don't existe already.
    Args:
        data_dict (Dict[str, pd.Dataframe]): Dictionary with key-value pairs for each data source.
    Returns:
        str: Message describing if data was inserted or not.
    """
    msg = {}
    x_new, y_new = prepare_data(data_dict)
    x, y = get_db_data()
    dates_x_new = [i for i in x_new['timestamp']
                   if i not in x['timestamp'].values]
    dates_y_new = [i for i in y_new['timestamp']
                   if i not in y['timestamp'].values]
    if dates_x_new and dates_y_new:
        x_in = x_new[x_new['timestamp'].isin(dates_x_new)]
        y_in = y_new[y_new['timestamp'].isin(dates_y_new)]
        create_db(mode='append', x_in=x_in, y_in=y_in)
        dates_x_new = [pd.to_datetime(d, unit='s').strftime(
            format='%Y-%m') for d in dates_x_new]
        dates_y_new = [pd.to_datetime(d, unit='s').strftime(
            format='%Y-%m') for d in dates_y_new]
        logger.info(
            f'Added feature records for months {", ".join(dates_x_new)}.')
        logger.info(
            f'Added target records for months {", ".join(dates_y_new)}.')
        msg = {'x': (f'Los datos de features se insertaron exitosamente para los meses {", ".join(dates_x_new)}.', 'success'),
               'y': (f'Los datos de target se insertaron exitosamente para los meses {", ".join(dates_y_new)}.', 'success')}
        db_data_span()
    elif dates_x_new:
        x_in = x_new[x_new['timestamp'].isin(dates_x_new)]
        create_db(mode='append', x_in=x_in, y_in=None)
        dates_x_new = [pd.to_datetime(d, unit='s').strftime(
            format='%Y-%m') for d in dates_x_new]
        dates_y_new = [pd.to_datetime(d, unit='s').strftime(
            format='%Y-%m') for d in dates_y_new]
        logger.info('Uplodaded feature records already exist in database.')
        msg = {'x': (f'Los datos de features se insertaron exitosamente para los meses {", ".join(dates_x_new)}.', 'success'),
               'y': ('Los datos de target a insertar ya existen en la base de datos.', 'info')}
        db_data_span()
    elif dates_y_new:
        y_in = y_new[y_new['timestamp'].isin(dates_y_new)]
        create_db(mode='append', x_in=None, y_in=y_in)
        dates_x_new = [pd.to_datetime(d, unit='s').strftime(
            format='%Y-%m') for d in dates_x_new]
        dates_y_new = [pd.to_datetime(d, unit='s').strftime(
            format='%Y-%m') for d in dates_y_new]
        logger.info('Uplodaded target records already exist in database.')
        msg = {'x': ('Los datos de features a insertar ya existen en la base de datos.', 'info'),
               'y': (f'Los datos de target se insertaron exitosamente para los meses {", ".join(dates_y_new)}.', 'success')}
        db_data_span()
    else:
        logger.info('Uplodaded records already exist in database.')
        msg = {'x': ('Los datos de features a insertar ya existen en la base de datos.', 'info'),
               'y': ('Los datos de target a insertar ya existen en la base de datos.', 'info')}
    return msg


def db_data_span() -> None:
    """
    Save all dates that are stored in the database as json file.
    """
    x, y = get_db_data()
    data = x.merge(y, how='inner', on='timestamp')
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    data = {'db_span': list(data.timestamp.apply(
        lambda x: f'{x.year}-{x.month:02d}'))}
    with open(os.path.join('logs', 'db_span.json'), 'w') as f:
        json.dump(data, f)
    logger.info(
        f"Updated DB available dates at {os.path.join('logs', 'db_span.json')}.")
