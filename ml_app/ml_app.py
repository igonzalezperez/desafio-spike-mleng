from flask.helpers import url_for
from joblib import load
import pandas as pd
from flask import Flask, request, jsonify, render_template, flash, redirect
from werkzeug.utils import secure_filename
from params import REQUIRED_COLUMNS
import secrets

app = Flask(__name__)
secret = secrets.token_urlsafe(32)
app.secret_key = secret

model = load('models/ridge_model.pkl')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/load_data', methods=['POST'])
def load_data():
    if request.method == 'POST':
        files = request.files.getlist("files[]")
        if len(files) < 3:
            flash(
                'Se necesitan 3 fuentes de datos', 'danger')
        elif not all([f.filename.endswith('.csv') for f in files]):
            flash("Los archivos deben ser .csv", 'danger')
        else:
            # Cargar datos a db
            flash('Datos cargados correctamente :)', 'success')
        return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    pred = 0
    return render_template('home.html', prediction_text=f'El precio de la leche es {pred}')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    pred = model.predict(data)
    return jsonify(pred)


@app.route('/logs')
def logs():
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
