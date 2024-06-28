FROM tiangolo/uvicorn-gunicorn:python3.11

LABEL maintainer="Ian <ianvzs@outlook.com>"

COPY app/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY ./app /app
