import streamlit as st
import sys
from streamlit import cli as stcli
from matplotlib.backends.backend_agg import RendererAgg
import plotly.express as px
import pandas as pd
from get_data import get_info


_lock = RendererAgg.lock
apptitle = 'Time series forecasting via topological data analysis'

st.set_page_config(page_title=apptitle, page_icon=":eyeglasses:", layout='wide')
st.title(apptitle)

col1, col2 = st.columns(2)


@st.cache
def load_data(dates):
    real_data = pd.read_csv('data/raw/data.csv', index_col='symbol')
    real_data.columns = [item.split()[0] for item in real_data.columns.to_list()]
    real_data = real_data[dates]
    return real_data


@st.cache
def load_prediction():
    prediction_data = pd.read_csv('data/prediction.csv', index_col='shortName')
    prediction_data.drop(
        ['2021-09-18', '2021-09-25', '2021-10-02', '2021-10-09', '2021-10-16', '2021-10-23', '2021-10-30', '2021-11-05'],
        axis=1,
        inplace=True
    )
    return prediction_data


preds = load_prediction()
real = load_data(preds.columns.to_list())


with st.spinner('Loading Forecasting...'):
    for i, item in enumerate(real.index):
        ticker_info = get_info(item, '1d', ['shortName'])
        data = pd.DataFrame({'Real values': real.loc[item], 'Predicted': preds.loc[item]})
        fig = px.line(
            data,
            labels={
                "index": "Date",
                "Value": "Close price"
            },
            title=f"{ticker_info['shortName']}"
        )
        if i % 2 == 0:
            col1.plotly_chart(fig)
        else:
            col2.plotly_chart(fig)


if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(stcli.main())
