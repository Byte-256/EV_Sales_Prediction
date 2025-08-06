# ev_forecast.py
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import itertools
import warnings

warnings.filterwarnings("ignore")

# Load and filter data
raw_data = pd.read_csv('./data/IEA_Global_EV_Data_2024.csv')
sales = raw_data[(raw_data['parameter'] == "EV sales") & (raw_data['category'] == "Historical")]

# Summarize total sales and regions
total_sales = int(sales['value'].sum())
num_regions = sales['region'].nunique()

# Yearly aggregation
yearly_sales = sales.groupby('year')['value'].sum().reset_index()
yearly_sales['value'] = yearly_sales['value'].astype(int)
yearly_sales = yearly_sales.sort_values('year').reset_index(drop=True)

# Prepare time series data
ts_data = yearly_sales.set_index('year')['value']
train_size = int(len(ts_data) * 0.7)
train_data, test_data = ts_data[:train_size], ts_data[train_size:]

# Stationarity check
adf_stat, p_value, _, _, critical_values, _ = adfuller(train_data.values)
stationarity_status = "Stationary" if p_value <= 0.05 else "Non-stationary"
if stationarity_status == "Non-stationary":
    train_data = train_data.diff().dropna()

# Grid search for ARIMA(p,d,q)
p, d, q = range(0, 4), range(0, 3), range(0, 4)
best_aic, best_order, best_model = np.inf, None, None

for order in itertools.product(p, d, q):
    try:
        model = ARIMA(train_data, order=order).fit()
        if model.aic < best_aic:
            best_aic, best_order, best_model = model.aic, order, model
    except:
        continue

# Evaluate model
# train_pred = best_model.fittedvalues
# train_resid = train_data - train_pred
# test_pred = best_model.forecast(steps=len(test_data))
# test_resid = test_data - test_pred

# train_metrics = {
#     "MSE": mean_squared_error(train_data, train_pred),
#     "MAE": mean_absolute_error(train_data, train_pred),
#     "MAPE": mean_absolute_percentage_error(train_data, train_pred),
# }

# test_metrics = {
#     "MSE": mean_squared_error(test_data, test_pred),
#     "MAE": mean_absolute_error(test_data, test_pred),
#     "MAPE": mean_absolute_percentage_error(test_data, test_pred),
# }

# Retrain on full data for future forecast
final_model = ARIMA(ts_data, order=best_order).fit()

arima_pred = final_model.predict(start=1, end=len(ts_data)-1, typ='levels')
arima_pred.index = ts_data.index[1:]

residuals = ts_data[1:] - arima_pred

# Build lag features from residuals
def create_lag_features(series, lags=3):
    df = pd.DataFrame({'residual': series})
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['residual'].shift(i)
    return df.dropna()

lagged_data = create_lag_features(residuals)
X = lagged_data.drop(columns='residual')
y = lagged_data['residual']

# Train Random Forest on residuals
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

forecast_steps = 5
last_known_year = ts_data.index[-1]
future_years = [last_known_year + i for i in range(1, forecast_steps + 1)]

arima_forecast = final_model.forecast(steps=forecast_steps)

last_resid = residuals[-3:].values.tolist()

rf_forecast = []
for _ in range(forecast_steps):
    X_input = np.array(last_resid[-3:]).reshape(1, -1)
    rf_pred = rf_model.predict(X_input)[0]
    rf_forecast.append(rf_pred)
    last_resid.append(rf_pred)  # update lag history

hybrid_forecast = arima_forecast.values + np.array(rf_forecast)

forecast_df = pd.DataFrame({
    'year': future_years,
    'ARIMA': arima_forecast.values,
    'RF_residual': rf_forecast,
    'Hybrid_Prediction': hybrid_forecast
})

if __name__ == "__main__":
    print("\n🔮 Hybrid Forecast (ARIMA + Random Forest):")
    print(forecast_df)
