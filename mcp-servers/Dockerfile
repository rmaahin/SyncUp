FROM python:3.12.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY google-calendar/server.py /app/google-calendar/server.py
COPY google-docs/server.py /app/google-docs/server.py
COPY github/server.py /app/github/server.py
