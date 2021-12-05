from joblib import load
import numpy as np
from data_science.preprocessing import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def load_model(filename):
    return load(filename)


model_0 = load_model('models/model_0.pkl')
model_1 = load_model('models/model_1.pkl')
data = preprocessing()
x = data.drop(['Precio_leche'], axis=1)
y = data['Precio_leche']
print('Model 1')
np.random.seed(0)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
preds = model_0.predict(x_test)
rmse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(rmse)
print(r2)

print('Model 2')
np.random.seed(0)
x = x[[i for i in list(x.columns) if not ('leche' in i)]]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

preds = model_1.predict(x_test)
rmse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(rmse)
print(r2)
breakpoint()
