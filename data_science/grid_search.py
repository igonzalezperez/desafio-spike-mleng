"""
DOCSTRING: TODO
"""
# %% Imports
import os
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from loguru import logger
from dotenv import load_dotenv
from logger_config import logger_config
from data_science.preprocessing import preprocessing

# %% Config
load_dotenv()
logger_config(level=os.getenv('LOG_LEVEL'))
pd.options.mode.chained_assignment = None  # default='warn'
plt.style.use('seaborn-notebook')
warnings.filterwarnings(
    "ignore", message="The localize method is no longer necessary, as this time zone supports the fold attribute",)

data = preprocessing()
X = data.drop(['Precio_leche'], axis=1)
y = data['Precio_leche']

logger.debug('Grid search')
np.random.seed(0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

pipe = Pipeline([('scale', StandardScaler()),
                ('selector', SelectKBest(mutual_info_regression)),
                ('poly', PolynomialFeatures()),
                ('model', Ridge())])
k = [3, 4, 5, 6, 7, 10]
alpha = [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
poly = [1, 2, 3, 5, 7]
grid = GridSearchCV(estimator=pipe,
                    param_grid=dict(selector__k=k,
                                    poly__degree=poly,
                                    model__alpha=alpha),
                    cv=3,
                    scoring='r2')
grid.fit(X_train, y_train)
joblib.dump(grid.best_estimator_, 'models/model_0.pkl')
y_predicted = grid.predict(X_test)

# evaluar modelo
rmse = mean_squared_error(y_test, y_predicted)
r2 = r2_score(y_test, y_predicted)

# printing values
logger.debug(f'RMSE: {rmse}')
logger.debug(f'R2: {r2}')
logger.debug(grid.best_params_)
# %%

# %%
X_train.columns[grid.best_estimator_.named_steps['selector'].get_support()]

# %%
predicted = pd.DataFrame(y_test).reset_index(drop=True)
predicted['predicc'] = y_predicted
predicted = predicted.reset_index()
predicted['residual'] = predicted['Precio_leche'] - predicted['predicc']

logger.debug('Grid search')
np.random.seed(0)
cols_no_leche = [x for x in list(X.columns) if not ('leche' in x)]
X_train = X_train[cols_no_leche]
X_test = X_test[cols_no_leche]

pipe = Pipeline([('scale', StandardScaler()),
                ('selector', SelectKBest(mutual_info_regression)),
                ('poly', PolynomialFeatures()),
                ('model', Ridge())])
k = [3, 4, 5, 6, 7, 10]
alpha = [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
poly = [1, 2, 3, 5, 7]
grid = GridSearchCV(estimator=pipe,
                    param_grid=dict(selector__k=k,
                                    poly__degree=poly,
                                    model__alpha=alpha),
                    cv=3,
                    scoring='r2')
grid.fit(X_train, y_train)
joblib.dump(grid.best_estimator_, './models/model_1.pkl')
y_predicted_noleche = grid.predict(X_test)

# evaluar modelo
rmse = mean_squared_error(y_test, y_predicted_noleche)
r2 = r2_score(y_test, y_predicted_noleche)

# printing values
logger.debug(f'RMSE: {rmse}')
logger.debug(f'R2: {r2}')
logger.debug(grid.best_params_)

# %%
X_train.columns[grid.best_estimator_.named_steps['selector'].get_support()]

predicted = pd.DataFrame(y_test).reset_index(drop=True)
predicted['predicc'] = y_predicted_noleche
predicted = predicted.reset_index()
predicted['residual'] = predicted['Precio_leche'] - predicted['predicc']
