"""
DOCSTRING: TODO
"""
# %% Imports
import os
import warnings
import json
import joblib
from loguru import logger
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from joblib import load
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt

from src.utils.logger_config import logger_config
from src.preprocessing import get_data, DataPreparation

# %% Config
load_dotenv()
logger_config(level=os.getenv('LOG_LEVEL'))
pd.options.mode.chained_assignment = None  # default='warn'
plt.style.use('seaborn-notebook')
warnings.filterwarnings(
    "ignore", message="The localize method is no longer necessary, as this time zone supports the fold attribute",)


def grid_search():
    x, y = get_data()
    x, y = DataPreparation().transform(x, y)

    logger.debug('Grid search')
    np.random.seed(0)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    k = [3, 4, 5, 6, 7, 10]
    alpha = [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    poly = [1, 2, 3, 5, 7]

    pipe = Pipeline([('scale', StandardScaler()),
                    ('selector', SelectKBest(mutual_info_regression)),
                    ('poly', PolynomialFeatures()),
                    ('model', Ridge())])
    grid = GridSearchCV(estimator=pipe,
                        param_grid=dict(selector__k=k,
                                        poly__degree=poly,
                                        model__alpha=alpha),
                        cv=3,
                        scoring='r2')
    grid.fit(x_train, y_train)
    y_predicted = grid.predict(x_test)

    # evaluar modelo
    rmse = mean_squared_error(y_test, y_predicted)
    r2 = r2_score(y_test, y_predicted)

    # printing values
    logger.debug(f'RMSE: {rmse}')
    logger.debug(f'R2: {r2}')
    logger.debug(grid.best_params_)
    with open('models/params/best_params.json', 'w') as f:
        json.dump(grid.best_params_, f)
    # # %%
    # predicted = pd.DataFrame(y_test).reset_index(drop=True)
    # predicted['predicc'] = y_predicted
    # predicted = predicted.reset_index()
    # predicted['residual'] = predicted['Precio_leche'] - predicted['predicc']

    # logger.debug('Grid search')
    # np.random.seed(0)
    # cols_no_leche = [x for x in list(x.columns) if not ('leche' in x)]
    # x_train = x_train[cols_no_leche]
    # x_test = x_test[cols_no_leche]

    # pipe = Pipeline([('scale', StandardScaler()),
    #                 ('selector', SelectKBest(mutual_info_regression)),
    #                 ('poly', PolynomialFeatures()),
    #                 ('model', Ridge())])

    # grid = GridSearchCV(estimator=pipe,
    #                     param_grid=dict(selector__k=k,
    #                                     poly__degree=poly,
    #                                     model__alpha=alpha),
    #                     cv=3,
    #                     scoring='r2')
    # grid.fit(x_train, y_train)
    # joblib.dump(grid.best_estimator_, './models/model_1.pkl')
    # y_predicted_noleche = grid.predict(x_test)

    # # evaluar modelo
    # rmse = mean_squared_error(y_test, y_predicted_noleche)
    # r2 = r2_score(y_test, y_predicted_noleche)

    # # printing values
    # logger.debug(f'RMSE: {rmse}')
    # logger.debug(f'R2: {r2}')
    # logger.debug(grid.best_params_)


def train():
    x, y = get_data()
    logger.debug('Train model')
    np.random.seed(0)
    x, y = DataPreparation().transform(x, y)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)
    with open('models/params/best_params.json', 'r') as f:
        params = json.load(f)

    preproc_pipe = Pipeline([('scale', StandardScaler()),
                             ('selector', SelectKBest(mutual_info_regression,
                                                      k=params['selector__k'])),
                             ('poly', PolynomialFeatures(degree=params['poly__degree']))])
    target_transform = StandardScaler()

    preproc_pipe.fit(x_train, y_train)
    target_transform.fit(y_train.values.reshape(-1, 1))

    x_train = preproc_pipe.transform(x_train)
    x_test = preproc_pipe.transform(x_test)
    y_train = target_transform.transform(y_train.values.reshape(-1, 1))
    y_test = target_transform.transform(y_test.values.reshape(-1, 1))

    model = Ridge(alpha=params['model__alpha'])
    model.fit(x_train, y_train)

    y_predicted = model.predict(x_test)

    y_test = target_transform.inverse_transform(y_test)
    y_predicted = target_transform.inverse_transform(y_predicted)
    rmse = mean_squared_error(y_test, y_predicted)
    r2 = r2_score(y_test, y_predicted)

    logger.debug(f'RMSE: {rmse}')
    logger.debug(f'R2: {r2}')

    joblib.dump(preproc_pipe, './models/pipelines/feature_pipeline.pkl')
    joblib.dump(target_transform, './models/pipelines/target_pipeline.pkl')
    joblib.dump(model, './models/ridge_model.pkl')
