FROM python:3.12.2-bookworm

EXPOSE 8050

RUN apt-get update
RUN pip install dash
RUN pip install dash_table
RUN pip install plotly.express
RUN pip install pandas
RUN pip install dash-bootstrap-components

COPY . /app

RUN chmod -R 555 /app

USER 1000

CMD python /app/app.py