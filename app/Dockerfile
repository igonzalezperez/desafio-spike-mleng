# syntax=docker/dockerfile:1
FROM python:3.8-slim-buster

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt

CMD gunicorn --bind 0.0.0.0:5000 run:app