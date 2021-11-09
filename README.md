# sp500_forecasting

### Time series forecasting via topological data analysis

Inspired by https://www.youtube.com/watch?v=ysHB9X0F9NQ

### How to run

1. Run MLFlow server

`
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1
`

2. Run Luigi daemon

`luigid`

3. Run monitoring infastructure

From /prometheus `docker build -t prometheus_simple . --no-cache`

From root `docker-compose up`

4. Finally rin streamlit app via

`streamlit run app.py`
