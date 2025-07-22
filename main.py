import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

raw_data = pd.read_csv('./data/IEA_Global_EV_Data_2024.csv')
print(raw_data.info())
print(raw_data.head(10))

print(raw_data.value_counts().__len__())
print(raw_data.columns)


sales = raw_data[raw_data['parameter'] == "EV sales"]
sales = sales[sales['category'] == "Historical"]
print("total rows in sales DF =",sales.value_counts().__len__())


# sales.to_csv("data/sales.csv")

print("Total EV Sales in past: ",sales['value'].sum().astype(int))
print(sales['region'].nunique())

yearly_sales = sales.groupby('year')['value'].sum().reset_index()
yearly_sales['value'] = yearly_sales['value'].astype(int)
print(yearly_sales)


import plotly.express as px

line_fig = px.line(yearly_sales, yearly_sales['year'], yearly_sales['value'], hover_data="value")
line_fig.show()

# plt.figure(figsize=(8,5))
# plt.plot(yearly_sales['year'], yearly_sales['value'], marker="o")
# plt.title('EV Sales per Year')
# plt.xlabel("Year")
# plt.ylabel("EV Sales")
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.show()


country_sales = sales.groupby('region')['value'].sum().reset_index()
country_sales.columns = ['Country', 'Total EV Sales']


sort = country_sales.sort_values(by='Total EV Sales', ascending=False)
plt.figure(figsize=(12, 10))
sns.barplot(
    data=sort,
    y='Country',
    x='Total EV Sales',
    palette='viridis'
)
plt.title('Total EV Sales by Country')
plt.xlabel('EV Sales (Total)')
plt.ylabel('Country')
plt.tight_layout()
plt.show()


fig = px.choropleth(
    country_sales,
    locations='Country',
    locationmode='country names',
    color='Total EV Sales',
    color_continuous_scale='Viridis',
    title='Global EV Sales by Country',
)

fig.update_layout(geo=dict(showframe=False, showcoastlines=False))
fig.show()

## Importing Machine Learning Library
# from sci-kit learn, import the train_test_split method for sliptting the dataset into two one for `Training the model` another one for `Testing the model`


# Time Series Analysis and Forecasting with ARIMA
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import itertools
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Sort data by year for proper time series analysis
yearly_sales = yearly_sales.sort_values('year').reset_index(drop=True)
print("\nTime Series Data:")
print(yearly_sales)

# Set year as index for time series analysis
ts_data = yearly_sales.set_index('year')['value']
print(f"\nTime Series Shape: {ts_data.shape}")
print(f"Time Series Index: {ts_data.index.tolist()}")

# Split data chronologically (not randomly for time series)
train_size = int(len(ts_data) * 0.7)
train_data = ts_data[:train_size]
test_data = ts_data[train_size:]

print(f"\nTrain data: {train_data.index.tolist()}")
print(f"Test data: {test_data.index.tolist()}")

# Check stationarity
print("\n" + "="*50)
print("STATIONARITY TEST")
print("="*50)

adfuller_result = adfuller(train_data.values)
print(f"ADF Statistic: {adfuller_result[0]:.4f}")
print(f"p-value: {adfuller_result[1]:.4f}")
print(f"Critical Values:")
for key, value in adfuller_result[4].items():
    print(f"\t{key}: {value:.3f}")

if adfuller_result[1] > 0.05:
    print("❌ Series is NOT stationary - differencing may be needed")
    stationarity_status = "Non-stationary"
else:
    print("✅ Series is stationary - no differencing needed")
    stationarity_status = "Stationary"

# Grid search for best ARIMA parameters
print("\n" + "="*50)
print("ARIMA MODEL SELECTION")
print("="*50)

p = range(0, 4)
d = range(0, 3)
q = range(0, 4)
pdq = list(itertools.product(p, d, q))

best_aic = np.inf
best_order = None
best_model = None
results = []

print("Searching for optimal ARIMA parameters...")
for order in pdq:
    try:
        model = ARIMA(train_data, order=order)
        fitted_model = model.fit()
        aic = fitted_model.aic
        results.append((order, aic))

        if aic < best_aic:
            best_aic = aic
            best_order = order
            best_model = fitted_model

    except Exception as e:
        continue

print(f"\n✅ Best ARIMA order: {best_order}")
print(f"✅ Best AIC: {best_aic:.4f}")
print(f"✅ Model Summary:")
print(best_model.summary())

# Make predictions on test data
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

# In-sample predictions
train_predictions = best_model.fittedvalues
train_residuals = train_data - train_predictions

# Out-of-sample predictions
test_predictions = best_model.forecast(steps=len(test_data))
test_residuals = test_data - test_predictions

# Calculate metrics
train_mse = mean_squared_error(train_data, train_predictions)
train_mae = mean_absolute_error(train_data, train_predictions)
train_mape = mean_absolute_percentage_error(train_data, train_predictions)

test_mse = mean_squared_error(test_data, test_predictions)
test_mae = mean_absolute_error(test_data, test_predictions)
test_mape = mean_absolute_percentage_error(test_data, test_predictions)

print(f"Training Metrics:")
print(f"  MSE: {train_mse:,.2f}")
print(f"  MAE: {train_mae:,.2f}")
print(f"  MAPE: {train_mape:.2%}")

print(f"\nTesting Metrics:")
print(f"  MSE: {test_mse:,.2f}")
print(f"  MAE: {test_mae:,.2f}")
print(f"  MAPE: {test_mape:.2%}")

# Future forecasting
print("\n" + "="*50)
print("FUTURE FORECASTING")
print("="*50)

# Retrain on full dataset for final forecasting
final_model = ARIMA(ts_data, order=best_order).fit()

# Forecast next 5 years
forecast_steps = 5
forecast = final_model.forecast(steps=forecast_steps)
forecast_ci = final_model.get_forecast(steps=forecast_steps).conf_int()

# Create forecast dataframe
future_years = range(ts_data.index[-1] + 1, ts_data.index[-1] + forecast_steps + 1)
forecast_df = pd.DataFrame({
    'year': future_years,
    'forecasted_sales': forecast.values,
    'lower_ci': forecast_ci.iloc[:, 0].values,
    'upper_ci': forecast_ci.iloc[:, 1].values
})

print("📈 EV Sales Forecast for Next 5 Years:")
print(forecast_df)

# Visualization
plt.figure(figsize=(15, 10))

# Plot historical data
plt.subplot(2, 2, 1)
plt.plot(ts_data.index, ts_data.values, 'o-', label='Historical', color='blue')
plt.plot(forecast_df['year'], forecast_df['forecasted_sales'], 'o-', label='Forecast', color='red')
plt.fill_between(forecast_df['year'], forecast_df['lower_ci'], forecast_df['upper_ci'],
                 alpha=0.3, color='red', label='Confidence Interval')
plt.title('EV Sales: Historical vs Forecast')
plt.xlabel('Year')
plt.ylabel('EV Sales')
plt.legend()
plt.grid(True)

# Plot residuals
plt.subplot(2, 2, 2)
plt.plot(train_data.index, train_residuals, 'o-', label='Train Residuals')
plt.plot(test_data.index, test_residuals, 'o-', label='Test Residuals')
plt.title('Model Residuals')
plt.xlabel('Year')
plt.ylabel('Residuals')
plt.legend()
plt.grid(True)

# Plot actual vs predicted
plt.subplot(2, 2, 3)
plt.plot(train_data.index, train_data.values, 'o-', label='Actual Train', color='blue')
plt.plot(train_data.index, train_predictions, 'o-', label='Predicted Train', color='orange')
plt.plot(test_data.index, test_data.values, 'o-', label='Actual Test', color='green')
plt.plot(test_data.index, test_predictions, 'o-', label='Predicted Test', color='red')
plt.title('Actual vs Predicted')
plt.xlabel('Year')
plt.ylabel('EV Sales')
plt.legend()
plt.grid(True)

# Plot forecast only
plt.subplot(2, 2, 4)
combined_years = list(ts_data.index) + list(forecast_df['year'])
combined_sales = list(ts_data.values) + list(forecast_df['forecasted_sales'])
plt.plot(ts_data.index, ts_data.values, 'o-', label='Historical', color='blue')
plt.plot(forecast_df['year'], forecast_df['forecasted_sales'], 'o-', label='Forecast', color='red')
plt.fill_between(forecast_df['year'], forecast_df['lower_ci'], forecast_df['upper_ci'],
                 alpha=0.3, color='red')
plt.title('EV Sales Forecast')
plt.xlabel('Year')
plt.ylabel('EV Sales')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ACTIONABLE INSIGHTS
print("\n" + "="*60)
print("🎯 ACTIONABLE INSIGHTS & RECOMMENDATIONS")
print("="*60)

# Calculate growth rates
historical_growth = []
for i in range(1, len(ts_data)):
    growth = ((ts_data.iloc[i] - ts_data.iloc[i-1]) / ts_data.iloc[i-1]) * 100
    historical_growth.append(growth)

avg_historical_growth = np.mean(historical_growth)
recent_growth = historical_growth[-3:] if len(historical_growth) >= 3 else historical_growth

# Forecast growth rates
forecast_growth = []
for i in range(1, len(forecast_df)):
    growth = ((forecast_df.iloc[i]['forecasted_sales'] - forecast_df.iloc[i-1]['forecasted_sales']) /
              forecast_df.iloc[i-1]['forecasted_sales']) * 100
    forecast_growth.append(growth)

print(f"📊 MARKET ANALYSIS:")
print(f"  • Current EV market size: {ts_data.iloc[-1]:,.0f} units")
print(f"  • Historical average growth rate: {avg_historical_growth:.1f}%")
print(f"  • Recent growth trend (last 3 years): {np.mean(recent_growth):.1f}%")
print(f"  • Forecasted growth rate: {np.mean(forecast_growth):.1f}%")
print(f"  • Market stationarity: {stationarity_status}")

print(f"\n🔮 FORECAST INSIGHTS:")
print(f"  • Expected sales by {forecast_df.iloc[-1]['year']}: {forecast_df.iloc[-1]['forecasted_sales']:,.0f} units")
print(f"  • Total forecasted sales (5 years): {forecast_df['forecasted_sales'].sum():,.0f} units")
print(f"  • Confidence interval range: {forecast_df['lower_ci'].min():,.0f} - {forecast_df['upper_ci'].max():,.0f}")

# Market segments analysis
print(f"\n🌍 REGIONAL OPPORTUNITIES:")
top_markets = sort.head(10)
print(f"  • Top 3 markets: {', '.join(top_markets.head(3)['Country'].tolist())}")
print(f"  • Emerging markets (positions 4-10): {', '.join(top_markets.iloc[3:10]['Country'].tolist())}")

# Investment recommendations
total_historical = ts_data.sum()
total_forecast = forecast_df['forecasted_sales'].sum()
market_expansion = ((total_forecast - total_historical) / total_historical) * 100

print(f"\n💡 STRATEGIC RECOMMENDATIONS:")
print(f"  1. MARKET EXPANSION: {market_expansion:.1f}% growth expected over next 5 years")
print(f"  2. INVESTMENT TIMING: {'Aggressive expansion' if avg_historical_growth > 20 else 'Steady growth strategy'}")
print(f"  3. RISK ASSESSMENT: Model accuracy (MAPE): {test_mape:.1%}")

if test_mape < 0.15:
    print(f"     ✅ High confidence forecast - suitable for strategic planning")
elif test_mape < 0.25:
    print(f"     ⚠️  Moderate confidence - consider additional factors")
else:
    print(f"     ❌ Low confidence - use with caution, consider ensemble methods")

print(f"\n🎯 ACTION ITEMS:")
print(f"  • Production Planning: Prepare for {forecast_df.iloc[-1]['forecasted_sales']:,.0f} units by {forecast_df.iloc[-1]['year']}")
print(f"  • Supply Chain: Scale up for {np.mean(forecast_growth):.1f}% annual growth")
print(f"  • R&D Investment: Focus on top markets and emerging opportunities")
print(f"  • Risk Management: Monitor model performance and update quarterly")

print("\n" + "="*60)
