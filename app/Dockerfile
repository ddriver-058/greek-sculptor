FROM python:3.12.2-bookworm

EXPOSE 8050

RUN apt-get update
RUN pip install dash
RUN pip install dash_table
RUN pip install plotly.express
RUN pip install pandas
RUN pip install dash-bootstrap-components

COPY . /app

CMD python /app/app.py