"""
Optimize training parameters with grid search and save optimal model and pipelines.
"""
# %% Imports
import json

import joblib
import numpy as np
from loguru import logger
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, train_test_split

from data_science.preprocessing import DataPreparation
from database.database import get_db_data
np.random.seed(0)


# %% Classes and functions
def grid_search() -> None:
    """
    Grid search best parameters using the following Pipeline:
    - DataPreparation - Compute moving averages features.
    - StandardScaler
    - SelectKBest -  k = [3, 4, 5, 6, 7, 10]
    - PlolynomialFeatures - poly = [1, 2, 3, 5, 7]
    - Ridge - alpha = [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    Best parameters are locally stored as json.
    """
    x, y = get_db_data()
    x, y = DataPreparation().transform(x, y)

    logger.info('Grid search')
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

    rmse = mean_squared_error(y_test, y_predicted)
    r2 = r2_score(y_test, y_predicted)

    logger.info(f'RMSE: {rmse}')
    logger.info(f'R2: {r2}')
    logger.info(grid.best_params_)

    with open('models/params/best_params.json', 'w') as f:
        json.dump(grid.best_params_, f)


def train() -> None:
    """
    Train model using best parameters obtained through grid search. 80-20 train-test split. Preprocess and model.pipelines are stored as .pkl files with fit parameters on training data.
    - DataPreparation - It's not saved since no .fit method is used.
    - StandardScaler + SelectKBest + PolynomialFeatures - Preprocess pipeline fit on train data and stored as .pkl.
    - Ridge - Model pipeline fit in train data and stored as .pkl.
    - StandardScaler - Preprocess pipeline for target data, fit on train and stored as .pkl.
    """
    x, y = get_db_data()

    logger.info('Train model')

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

    logger.info(f'RMSE: {rmse}')
    logger.info(f'R2: {r2}')

    joblib.dump(preproc_pipe, './models/pipelines/feature_pipeline.pkl')
    joblib.dump(target_transform, './models/pipelines/target_pipeline.pkl')
    joblib.dump(model, './models/ridge_model.pkl')
