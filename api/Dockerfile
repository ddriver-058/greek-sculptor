FROM python:3.12.2-bookworm

EXPOSE 5000

RUN apt-get update
RUN pip install flask
RUN pip install pulp
RUN pip install pandas
RUN pip install scipy
RUN pip install yahooquery
RUN pip install QuantLib

COPY . /api

RUN chmod -R 555 /api

USER 1000

WORKDIR /api

CMD python /api/api.py