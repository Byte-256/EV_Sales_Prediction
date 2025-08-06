# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import forecast_df, yearly_sales, total_sales, num_regions

st.set_page_config(page_title="EV Sales Forecast", layout="centered")

st.title("🚗 Global EV Sales Forecast (Hybrid ARIMA + Random Forest)")
st.markdown("This dashboard uses a hybrid model to forecast future global EV sales.")

# Overview
st.subheader("🔢 Dataset Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"{total_sales:,}")
col2.metric("Years", yearly_sales['year'].nunique())
col3.metric("Regions", num_regions)

# Historical data
st.subheader("📈 Historical Sales")
fig_hist, ax = plt.subplots()
sns.lineplot(data=yearly_sales, x='year', y='value', marker='o', ax=ax)
ax.set_title("EV Sales Over Time")
ax.set_ylabel("Units Sold")
st.pyplot(fig_hist)

# Forecast
st.subheader("🔮 Forecast (Next 5 Years)")
st.dataframe(forecast_df.set_index('year'), use_container_width=True)

fig_pred, ax = plt.subplots()
ax.plot(yearly_sales['year'], yearly_sales['value'], label="Historical", marker='o')
ax.plot(forecast_df['year'], forecast_df['Hybrid_Prediction'], label="Forecast", marker='o', linestyle='--')
ax.set_title("EV Sales Forecast")
ax.set_ylabel("Units Sold")
ax.legend()
st.pyplot(fig_pred)

# Footer
st.markdown("---")
st.caption("Built using ARIMA + Random Forest Hybrid | Streamlit + Matplotlib")

