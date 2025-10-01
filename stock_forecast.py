# stock_forecast_auto_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA

from datetime import timedelta

st.set_page_config(page_title="Stock Forecasting - Fixed", layout="wide")
st.title(" Stock Forecasting ")

# ---------- Helpers ----------
def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns produced by yfinance into single-level names."""
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            # Join non-empty parts with underscore, e.g. ('Close','AAPL') -> 'Close_AAPL'
            parts = [str(c).strip() for c in col if c not in (None, '') and str(c).strip() != '']
            new = "_".join(parts) if parts else ""
            new_cols.append(new)
        df.columns = new_cols
    else:
        df.columns = [str(c) for c in df.columns]
    return df

def get_close_columns(df: pd.DataFrame):
    """Return list of candidate close columns (case-insensitive 'close' prefix)."""
    return [c for c in df.columns if c.lower().startswith("close")]

def make_forecast_arima(series: pd.Series, horizon: int):
    model = ARIMA(series, order=(5,1,0))
    fit = model.fit()
    pred = fit.get_forecast(steps=horizon).predicted_mean
    last_date = series.index.max()
    dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
    return pd.DataFrame({"Date": dates, "Forecast": pred.values})



def make_demo_lstm_forecast(series: pd.Series, horizon: int):
    # Simple upward linear demo (replace with full LSTM pipeline if desired)
    last = series.iloc[-1]
    future_vals = np.linspace(last, last * 1.05, horizon)
    dates = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=horizon, freq='D')
    return pd.DataFrame({"Date": dates, "Forecast": future_vals})

# ---------- UI ----------
tickers_input = st.text_input("Tickers (comma-separated) — e.g. AAPL,MSFT,TSLA", value="AAPL")
horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=90, value=30)
model_choice = st.selectbox("Model", ["ARIMA", "Prophet", "Demo LSTM"])

if st.button("Run Forecast"):
    try:
        tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
        # yfinance: pass list for multiple; it returns MultiIndex columns in that case
        raw = yf.download(tickers, start="2018-01-01", progress=False)
        raw = raw.reset_index()  # bring Date out of index if present
        # flatten column names
        raw = flatten_cols(raw)
        # ensure Date column named 'Date'
        date_col_candidates = [c for c in raw.columns if c.lower() == 'date']
        if not date_col_candidates:
            st.error("Couldn't find a Date column in downloaded data.")
            st.stop()
        date_col = date_col_candidates[0]
        raw.rename(columns={date_col: "Date"}, inplace=True)
        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
        raw = raw.dropna(subset=["Date"]).set_index("Date").sort_index()

        st.success("Data downloaded and flattened successfully.")
        st.write("Available columns:", list(raw.columns))

        # find close columns (e.g., 'Close' or 'Close_AAPL')
        close_cols = get_close_columns(raw)
        if not close_cols:
            st.error("No 'Close' columns detected in the data.")
            st.stop()

        # if multiple close columns, let user pick target ticker to forecast
        selected_close = st.selectbox("Select Close column to forecast", close_cols, index=0)

        # show interactive history of closes (all close_* for comparison)
        close_compare = raw[[c for c in close_cols if c in raw.columns]].reset_index().melt(id_vars="Date",
                                                                                          var_name="Ticker",
                                                                                          value_name="Close")
        fig_hist = px.line(close_compare, x="Date", y="Close", color="Ticker", title="Historical Close Prices (compare)")
        st.plotly_chart(fig_hist, use_container_width=True)

        # prepare series to forecast
        series = raw[[selected_close]].dropna().squeeze()
        series.index = pd.to_datetime(series.index)
        series.name = "Close"

        # compute forecast
        if model_choice == "ARIMA":
            fut_df = make_forecast_arima(series, horizon)
        
        else:
            fut_df = make_demo_lstm_forecast(series, horizon)

        # combine and show
        fut_df["Date"] = pd.to_datetime(fut_df["Date"])
        combined = pd.concat([series.reset_index().rename(columns={series.name:"Actual"}), 
                              fut_df.rename(columns={"Forecast":"Forecast"})], sort=False, ignore_index=True)

        # plot actual + forecast (overlay)
        plot_df_actual = series.reset_index().rename(columns={series.name:"Value"})
        plot_df_forecast = fut_df.rename(columns={"Forecast":"Value"})
        plot_df_forecast["Type"] = "Forecast"
        plot_df_actual["Type"] = "Actual"
        plot_all = pd.concat([plot_df_actual, plot_df_forecast], ignore_index=True)
        fig = px.line(plot_all, x="Date", y="Value", color="Type", title=f"{selected_close} - Actual vs Forecast")
        st.plotly_chart(fig, use_container_width=True)

        # export CSV for Power BI
        out_df = pd.concat([
            plot_df_actual.rename(columns={"Value":"Actual"}).set_index("Date"),
            plot_df_forecast.rename(columns={"Value":"Forecast"}).set_index("Date")
        ], axis=1).reset_index()
        st.download_button(" Export CSV for Power BI", out_df.to_csv(index=False).encode("utf-8"), f"{tickers_input}_forecast.csv", "text/csv")
        st.success("Forecast complete — export ready for Power BI.")

    except Exception as e:
        st.error(f"Failed: {e}")
        st.exception(e)
