from typing import Tuple
import pandas as pd
from loguru import logger
import sqlalchemy as db

from data_science.preprocessing import ingest_data
from data_science.preprocessing import prepare_data


def create_db(mode: str = 'fail', x_in: pd.DataFrame = None, y_in: pd.DataFrame = None) -> None:
    """
    Create an sqlite databse using the original .csv data.

    Args:
        mode (str, optional): What to do if DB already exists, it can be set to 'replace' to reset DB or to 'append' to add more rows. Defaults to 'fail'.
        x_in (pd.DataFrame, optional): Feature data to be inserted if mode is set to 'append'. Defaults to None.
        y_in (pd.DataFrame, optional): Target data to be inserted if mode is set to 'append'. Defaults to None.
    """
    engine = db.create_engine('sqlite:///database/database.db', echo=False)
    conn = engine.connect()
    if mode == 'append':
        x_in.to_sql(name='features', con=conn,
                    if_exists=mode, index=False)
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
            "Tables 'features' and 'target' already exist. Set 'mode='replace'' to overwrite.")
    conn.close()
    engine.dispose()
    return


def get_db_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch data from the database's tables (features and target).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Data for features and target from de database as DataFrames.
    """
    engine = db.create_engine('sqlite:///database/database.db', echo=False)
    conn = engine.connect()
    x = pd.read_sql_table('features', conn)
    y = pd.read_sql_table('target', conn)
    conn.close()
    engine.dispose()
    return x, y


def insert_rows(data_dict):
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
        logger.info(f'Added feature records for months {dates_x_new}.')
        logger.info(f'Added target records for months {dates_y_new}.')
        msg = {'x': (f'Los datos de features se insertaron exitosamente para los meses {dates_x_new}.', 'success'),
               'y': (f'Los datos de target se insertaron exitosamente para los meses {dates_y_new}.', 'success')}
    elif dates_x_new:
        x_in = x_new[x_new['timestamp'].isin(dates_x_new)]
        dates_x_new = [pd.to_datetime(d, unit='s').strftime(
            format='%Y-%m') for d in dates_x_new]
        dates_y_new = [pd.to_datetime(d, unit='s').strftime(
            format='%Y-%m') for d in dates_y_new]
        logger.info('Uplodaded feature records already exist in database.')
        msg = {'x': (f'Los datos de features se insertaron exitosamente para los meses {dates_x_new}.', 'success'),
               'y': ('Los datos de target a insertar ya existen en la base de datos.', 'danger')}
    elif dates_y_new:
        y_in = y_new[y_new['timestamp'].isin(dates_y_new)]
        dates_x_new = [pd.to_datetime(d, unit='s').strftime(
            format='%Y-%m') for d in dates_x_new]
        dates_y_new = [pd.to_datetime(d, unit='s').strftime(
            format='%Y-%m') for d in dates_y_new]
        logger.info('Uplodaded target records already exist in database.')
        msg = {'x': ('Los datos de features a insertar ya existen en la base de datos.', 'danger'),
               'y': (f'Los datos de target se insertaron exitosamente para los meses {dates_y_new}.', 'success')}
    else:
        logger.info('Uplodaded records already exist in database.')
        msg = {'x': ('Los datos de features a insertar ya existen en la base de datos.', 'danger'),
               'y': ('Los datos de target a insertar ya existen en la base de datos.', 'danger')}
    return msg


if __name__ == '__main__':
    create_db(mode='replace')
