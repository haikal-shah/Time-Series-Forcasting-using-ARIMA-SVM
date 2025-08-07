import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

st.set_page_config(page_title="Electric Load Forecasting", layout="centered")

st.title("Date Time Forecasting Using ARIMA + SVM")
st.write("Upload your CSV file containing a datetime column and at least one numeric column.")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Columns in uploaded file:", df.columns.tolist())

        df.columns = df.columns.str.strip()

        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "datetime"})

        datetime_col = st.selectbox("Select the datetime column", df.columns)
        target_col = st.selectbox("Select the load column (target for forecasting)", df.select_dtypes(include=[np.number]).columns)

        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        df = df[[datetime_col, target_col]].dropna()
        df.set_index(datetime_col, inplace=True)
        df.sort_index(inplace=True)

        st.subheader("Data Preview")
        preview_options = [5, 10, 20, 50, 100, 'All']
        preview_rows = st.selectbox("Rows to preview", preview_options, index=0)
        st.dataframe(df if preview_rows == 'All' else df.head(preview_rows))

        st.success("Data processed successfully.")
        st.line_chart(df[target_col], use_container_width=True)

        forecast_unit = st.radio("Select forecast unit", ["Hours", "Days"])
        max_horizon = 168 if forecast_unit == "Hours" else 30
        forecast_horizon = st.slider(f"Forecast Horizon ({forecast_unit.lower()})", 1, max_horizon, 24)

        # Determine steps and future timestamps
        freq = pd.infer_freq(df.index)
        if not freq:
            freq = pd.infer_freq(df.index[:10])
        time_delta = timedelta(hours=1) if forecast_unit == "Hours" else timedelta(days=1)
        future_dates = [df.index[-1] + time_delta * i for i in range(1, forecast_horizon + 1)]

        # ARIMA Forecast
        arima_model = ARIMA(df[target_col], order=(5, 1, 0))
        arima_result = arima_model.fit()
        arima_forecast = arima_result.forecast(steps=forecast_horizon)
        arima_forecast = pd.Series(arima_forecast.values, index=future_dates)

        # SVM Forecast
        df_svm = df.copy()
        df_svm['ordinal'] = df_svm.index.map(lambda x: x.toordinal())
        df_svm['hour'] = df_svm.index.hour
        df_svm['dayofweek'] = df_svm.index.dayofweek

        X = df_svm[['ordinal', 'hour', 'dayofweek']]
        y = df_svm[target_col]

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

        svm = SVR()
        svm.fit(X_scaled, y_scaled)

        future_df = pd.DataFrame({
            'ordinal': [d.toordinal() for d in future_dates],
            'hour': [d.hour for d in future_dates],
            'dayofweek': [d.weekday() for d in future_dates]
        })

        future_X_scaled = scaler_X.transform(future_df)
        svm_forecast_scaled = svm.predict(future_X_scaled)
        svm_forecast = scaler_y.inverse_transform(svm_forecast_scaled.reshape(-1, 1)).ravel()

        # Combine forecasts
        forecast_df = pd.DataFrame({
            'ARIMA Forecast': arima_forecast,
            'SVM Forecast': svm_forecast
        }, index=future_dates)

        st.subheader("Forecast Results")
        st.line_chart(forecast_df)
        st.dataframe(forecast_df)

        st.download_button(
            label="Download Forecast CSV",
            data=forecast_df.to_csv().encode('utf-8'),
            file_name="forecast_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to begin.")
