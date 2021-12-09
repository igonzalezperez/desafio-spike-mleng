from joblib import load
import pandas as pd
from flask import Flask, request, render_template, flash
from dateutil.relativedelta import relativedelta as rdelta
from params import REQUIRED_COLUMNS
import secrets
from src.predict import make_predictions
from src.preprocessing import get_data, DataPreparation

app = Flask(__name__)
secret = secrets.token_urlsafe(32)
app.secret_key = secret


def parse_dates(dates: str):
    try:
        if ' ' in dates:
            dates = dates.split(' ')
            date_i = pd.to_datetime(dates[0], format='%Y-%m')
            date_f = pd.to_datetime(dates[1], format='%Y-%m')
            months = (date_f.year - date_i.year) * 12 + \
                (date_f.month - date_i.month) + 1
            dates = [date_i + rdelta(months=i) for i in range(months)]
        elif ',' in dates:
            dates = dates.split(',')
            dates = [pd.to_datetime(d, format='%Y-%m') for d in dates]
        elif not dates:
            return None
        else:
            dates = [pd.to_datetime(dates, format='%Y-%m')]
    except:
        return None
    return dates


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
            # Cargar datos a db
            flash('Datos cargados correctamente :)', 'success')
        return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        dates = request.form['dates']
        dates = parse_dates(dates)
        if not dates:
            flash(
                'La fecha a predecir no corresponde a ningún formato aceptado.', 'danger')
            return render_template('home.html')
        preds, msg = make_predictions(dates)
        if isinstance(msg, str):
            flash(msg, 'danger')
            render_template('home.html')
        preds = preds.round(2)
        return render_template('home.html',  column_names=preds.columns.values, row_data=list(preds.values.tolist()),
                               table_name="Prices", zip=zip)


@app.route('/logs')
def logs():
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
