"""
Flask app for loading data, make batch predictions and monitoring.
"""
# %% Imports
import os
import json
from typing import List
import pandas as pd
from loguru import logger
from flask import Flask, request, render_template, flash, send_from_directory, abort
from dateutil.relativedelta import relativedelta as rdelta
import secrets
import socket

from werkzeug.utils import redirect
from database.database import db_data_span
from database.database import create_db
from data_science.predict import make_predictions
from database.database import insert_rows
import config.config as cfg

# %% Config
app = Flask(__name__)
secret = secrets.token_urlsafe(32)
app.secret_key = secret
app.config["LOGS"] = os.path.join(os.getcwd(), 'logs')

# %% Functions and Classes


def parse_dates(dates: str) -> List[pd.Timestamp]:
    """
    Parse dates recieved from /predict POST request. Behavior depends on the input:
    'YYYY-MM': Single date specifying year and month, separated by dash.
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
            if set(cfg.REQUIRED_COLUMNS['rain']).issubset(df.columns) and df[cfg.REQUIRED_COLUMNS['rain']].isna().sum().sum() == 0:
                data_dict['rain'] = df
            else:
                logger.error('Missing required columns.')
                return None, 'Faltan datos en las columnas requeridas'
        elif 'Periodo' in df.columns:
            if set(cfg.REQUIRED_COLUMNS['central_bank']).issubset(df.columns) and df[cfg.REQUIRED_COLUMNS['central_bank']].isna().sum().sum() == 0:
                data_dict['central_bank'] = df
            else:
                logger.error('Missing required columns.')
                return None, 'Faltan datos en las columnas requeridas'
        elif ('Anio' in df.columns) and ('Mes' in df.columns):
            if set(cfg.REQUIRED_COLUMNS['milk_price']).issubset(df.columns) and df[cfg.REQUIRED_COLUMNS['milk_price']].isna().sum().sum() == 0:
                data_dict['milk_price'] = df
            else:
                logger.error('Missing required columns')
                return None, 'Faltan datos en las columnas requeridas'
    if set(cfg.REQUIRED_COLUMNS.keys()) != set(data_dict.keys()):
        logger.error('Missing required data source.')
        return None, 'Faltan fuentes de datos requeridas.'
    return data_dict, None


@app.route('/')
def home():
    return render_template('home.html', container_id={'container_id': socket.gethostname()})


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
                make_predictions(pred_dates=None)
            else:
                flash(msg, 'danger')
        return redirect('/')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        dates = request.form['dates'].strip().strip('\'').strip('\"')
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


@app.route('/db_info')
def db_info():
    with open(os.path.join('logs', 'db_span.json'), 'r') as f:
        db_span = json.load(f)
    return render_template('db_info.html', db_span=db_span)


@app.route('/logs/train')
def train_logs():
    with open(os.path.join('logs', 'train.log'), 'r') as f:
        logs = f.readlines()
    return render_template('logs.html', logs=logs, title='Entrenamiento', fn='train.log')


@app.route('/logs/pred')
def pred_logs():
    with open(os.path.join('logs', 'predict.log'), 'r') as f:
        logs = f.readlines()
    return render_template('logs.html', logs=logs, title='Predicción', fn='predict.log')


@app.route('/reset-db')
def reset_db():
    logger.warning('Reset DB.')
    create_db(mode='replace')
    db_data_span()
    flash('La base de datos fue reseteada.', 'warning')
    return redirect('/')


@app.route('/get-logs/<log_name>', methods=['POST'])
def get_logs(log_name):
    if request.method == 'POST':
        try:
            return send_from_directory(app.config['LOGS'], log_name, as_attachment=True)
        except FileNotFoundError:
            abort(404)
