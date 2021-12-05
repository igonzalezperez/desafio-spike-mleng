from flask import Flask, request, jsonify, render_template
from joblib import load

app = Flask(__name__)
model = load('models/model_0.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/inference', methods=['POST'])
def inference():
    features = request.form.values()
    pred = model.predict(features)
    return render_template('index.html', prediction_text=f'Milk price is {pred}')


@app.route('/inference_api', methods=['POST'])
def inference_api():
    data = request.get_json(force=True)
    pred = model.predict(data)
    return jsonify(pred)


if __name__ == '__main__':
    app.run(debug=True)
