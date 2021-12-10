# # syntax=docker/dockerfile:1
FROM python:3.8-slim-buster

WORKDIR /app

ADD . /app

ENV LOG_LEVEL=INFO
ENV DEBUG=False
ENV DB_FOLDER=data
ENV DB_FILE=database.db
ENV MODEL_DIR=models
ENV PIPELINE_DIR=pipelines
ENV PARAM_DIR=params
ENV BEST_PARAMS_FILE=best_params.json
ENV FEATURE_PIPELINE_FILE=feature_pipeline.pkl
ENV TARGET_PIPELINE_FILE=target_pipeline.pkl
ENV MODEL_FILE=ridge_model.pkl

RUN pip install -r requirements.txt

CMD gunicorn --bind 0.0.0.0:5000 run:app