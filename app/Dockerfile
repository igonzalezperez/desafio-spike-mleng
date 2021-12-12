# # syntax=docker/dockerfile:1
FROM python:3.8-slim-buster

WORKDIR /app

ADD . /app

ENV LOG_LEVEL=INFO
ENV DEBUG=False
ENV DB_PATH=/data/sql/database.db
ENV BEST_PARAMS_PATH=/models/params/best_params.json
ENV FEATURE_PIPELINE_PATH=/models/pipelines/feature_pipeline.pkl
ENV TARGET_PIPELINE_PATH=/models/pipelines/target_pipeline.pkl
ENV MODEL_PATH=/models/ridge_model.pkl

RUN pip install -r requirements.txt
RUN python optimize.py

CMD gunicorn --bind 0.0.0.0:5000 run:app