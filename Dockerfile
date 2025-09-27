FROM python:3.11-slim

RUN pip install --upgrade pip

RUN mkdir -p /app

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt
