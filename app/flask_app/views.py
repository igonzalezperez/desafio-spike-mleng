"""
Flask app for loading data, make batch predictions and monitoring.
"""
# %% Imports
from typing import List
import pandas as pd
from loguru import logger
from flask import Flask, request, render_template, flash
from dateutil.relativedelta import relativedelta as rdelta
import secrets

from werkzeug.utils import redirect
from data_science.predict import make_predictions
from database.database import insert_rows
from params import REQUIRED_COLUMNS

# %% Config
app = Flask(__name__)
secret = secrets.token_urlsafe(32)
app.secret_key = secret


# %% Functions and Classes

def parse_dates(dates: str) -> List[pd.Timestamp]:
    """
    Parse dates recieved from /predict POST request. Behavior depends on the input:
    'YYYY-MM': Single date specifying year and month, separated by dash.
    'YYYY-MM,YYYY-MM,...': Same format as before but with multiple dates separated by a comma. In this cases each date is returned as datetime.
    'YYYY-MM YYYY-MM': Dates separated by a space. A list of all the datetimes for months between first and second date, including the borders. 
    If input doesn't conform with any of these formats return None.

    Args:
        dates (str): String with required dates for prediction.

    Returns:
        List[pd.Timestamp]: List of timestamps with the required prediction dates.
    """
    msg = None
    try:
        if ' ' in dates:
            dates = dates.split(' ')
            date_i = pd.to_datetime(dates[0], format='%Y-%m')
            date_f = pd.to_datetime(dates[1], format='%Y-%m')
            months = (date_f.year - date_i.year) * 12 + \
                (date_f.month - date_i.month) + 1
            dates = [date_i + rdelta(months=i) for i in range(months)]
        else:
            dates = [pd.to_datetime(dates, format='%Y-%m')]
    except ValueError:
        msg = 'La(s) fecha(s) a predecir no corresponde(n) a ningún formato aceptado.'
        logger.error("Value doesn't conform with accepted formats.")
    return dates, msg


def files_to_dict(files):
    data_dict = {}
    for f in files:
        df = pd.read_csv(f)
        if 'date' in df.columns:
            if set(REQUIRED_COLUMNS['rain']).issubset(df.columns) and df[REQUIRED_COLUMNS['rain']].isna().sum().sum() == 0:
                data_dict['rain'] = df
            else:
                logger.error('Missing required columns.')
                return None, 'Faltan datos en las columnas requeridas'
        elif 'Periodo' in df.columns:
            if set(REQUIRED_COLUMNS['central_bank']).issubset(df.columns) and df[REQUIRED_COLUMNS['central_bank']].isna().sum().sum() == 0:
                data_dict['central_bank'] = df
            else:
                logger.error('Missing required columns.')
                return None, 'Faltan datos en las columnas requeridas'
        elif ('Anio' in df.columns) and ('Mes' in df.columns):
            if set(REQUIRED_COLUMNS['milk_price']).issubset(df.columns) and df[REQUIRED_COLUMNS['milk_price']].isna().sum().sum() == 0:
                data_dict['milk_price'] = df
            else:
                logger.error('Missing required columns')
                return None, 'Faltan datos en las columnas requeridas'
    if set(REQUIRED_COLUMNS.keys()) != set(data_dict.keys()):
        logger.error('Missing required data source.')
        return None, 'Faltan fuentes de datos requeridas.'
    return data_dict, None


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/insert_data', methods=['POST'])
def insert_data():
    if request.method == 'POST':
        files = request.files.getlist("files[]")
        if len(files) < 3:
            flash(
                'Se necesitan 3 fuentes de datos (Precipitación - Índices económicos - Precio de la leche).', 'danger')
        elif not all([f.filename.endswith('.csv') for f in files]):
            flash("Los archivos deben ser .csv", 'danger')
        else:
            dd, msg = files_to_dict(files)
            if dd is not None:
                msg = insert_rows(dd)
                flash(*msg['x'])
                flash(*msg['y'])
            else:
                flash(msg, 'danger')
        return redirect('/')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        dates = request.form['dates']
        if not dates:
            return redirect('/')
        dates, msg = parse_dates(dates)
        if msg:
            flash(msg, 'danger')
            return redirect('/')
        preds, msg = make_predictions(dates)
        if isinstance(msg, str):
            flash(msg, 'danger')
            return redirect('/')
        flash('Predicción exitosa :)', 'success')
        preds = preds.round(2)
        return render_template('home.html',  column_names=preds.columns.values, row_data=list(preds.values.tolist()),
                               table_name="Prices", zip=zip)


@app.route('/monitor')
def monitor():
    return redirect('/')
