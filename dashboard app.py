import streamlit as st
import pandas as pd
from src.data_loader import load_stock_data
from src.preprocessing import preprocess
from src.arima_model import train_arima, forecast_arima

st.title("📈 Stock Market Forecasting Dashboard")

df = load_stock_data("../data/nifty.csv")
df = preprocess(df)

st.subheader("Close Price Chart")
st.line_chart(df['Close'])

if st.button("Run ARIMA Forecast"):
    model = train_arima(df)
    pred = forecast_arima(model, 30)
    st.subheader("Next 30 Days Forecast")
    st.line_chart(pred)