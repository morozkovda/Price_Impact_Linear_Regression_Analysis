import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
import time
import plotly.graph_objs as go
import plotly.express as px
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta

# df_PI = pd.read_csv(r'data/wbtc_usdc_PI new.csv')
# df_PI = pd.read_csv(r'data/usdc_wbtc_PI.csv')
# usdc_tvl = pd.read_csv(r'data/usdc_tvl_data_new.csv')
# wbtc_tvl = pd.read_csv(r'data/wbtc_tvl_data_new.csv')
# wbtcusdc_tvl = pd.read_csv(r'data/wbtcusdc_tvl_data_new.csv')

df_PI = pd.read_csv(r'data/usdc_weth_PI.csv')
usdc_tvl = pd.read_csv(r'data/usdc_tvl_data_new.csv')
wbtc_tvl = pd.read_csv(r'data/matic_tvl_data_new.csv')
wbtcusdc_tvl = pd.read_csv(r'data/wbtcusdc_tvl_data_new.csv')

df_PI = df_PI.drop(df_PI.columns[0], axis=1)
df_PI = df_PI.drop(df_PI.columns[3], axis=1)
df_PI = pd.melt(df_PI, id_vars=df_PI.columns[[0,1,2]], var_name='Position Size', value_name='PI')
df_PI[df_PI.columns[[0,1,2,3]]] = df_PI[df_PI.columns[[0,1,2,3]]].apply(lambda x: x.str.replace("]", ''))
df_PI[df_PI.columns[[0,1,2,3]]] = df_PI[df_PI.columns[[0,1,2,3]]].apply(lambda x: x.str.replace("'", ''))
df_PI.rename(columns={
    df_PI.columns[0]: "date",
    df_PI.columns[1]: "time",
    df_PI.columns[2]: "timestamp"},inplace= True)
print(df_PI.isna().any())

def try_convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        return float(value.replace("]", ''))


def try_convert_to_int(value):
    try:
        return int(value)
    except ValueError:
        return int(value.replace(' ', ''))

df_PI['PI'] = df_PI['PI'].apply(try_convert_to_float)
df_PI['Position Size'] = df_PI['Position Size'].apply(try_convert_to_int)

df_PI['datetime'] = df_PI['date'] + ' ' + df_PI['time']

df_PI['datetime'] = pd.to_datetime(df_PI['datetime'])
df_PI['datetime'] = df_PI['datetime'].apply(lambda dt: dt.replace(minute=0, second=0) if dt.minute < 30 else dt.replace(minute=0, second=0) + timedelta(hours=1))
df_PI['date'] = df_PI['datetime']
df_PI = df_PI.drop(['time','timestamp','datetime'],axis = 1)

df_PI = df_PI.groupby(['date','Position Size']).mean()
df_PI = df_PI.reset_index()

usdc_tvl = usdc_tvl[['date', 'tvl usd value']]
wbtc_tvl = wbtc_tvl[['date', 'tvl usd value']]
wbtcusdc_tvl = wbtcusdc_tvl[['date', 'tvl usd value']]
for i in [usdc_tvl, wbtc_tvl, wbtcusdc_tvl]:
    i['date'] = pd.to_datetime(i['date'])

df_merged = pd.merge(df_PI, usdc_tvl, on='date')
df_merged = pd.merge(df_merged, wbtc_tvl, on='date')
df_merged = pd.merge(df_merged, wbtcusdc_tvl, on='date')

# X_poly = sm.add_constant(df_merged[['Position Size','tvl usd value_x','tvl usd value_y', 'tvl usd value']])
# mod_poly = sm.OLS(df_merged['PI'], X_poly)
# reg_poly = mod_poly.fit()
# reg_poly.summary()

X_poly = sm.add_constant(df_merged[['Position Size','tvl usd value_x','tvl usd value_y']])
mod_poly = sm.OLS(df_merged['PI'], X_poly)
reg_poly = mod_poly.fit()
print(reg_poly.summary())


predictions = reg_poly.predict(X_poly)

fig = go.Figure()
trace1 = go.Scatter(y=df_merged['PI'], mode='lines', name='PI')
trace2 = go.Scatter(y=predictions, mode='lines', name='predictions')
fig = go.Figure(data=[trace1, trace2])
fig.show()


def evaluate_model(actual, predicted):
    metrics = {}
    # calculate mean squared error (MSE)
    mse = mean_squared_error(actual, predicted)
    metrics['MSE'] = mse
    # Calculate the RMSE using scikit-learn's mean_squared_error function
    rmse = np.sqrt(mse)
    metrics['RMSE'] = rmse
    # calculate mean absolute error (MAE)
    mae = mean_absolute_error(actual, predicted)
    metrics['MAE'] = mae
    for metric, value in metrics.items():
        print(f'{metric}: {value}')


evaluate_model(df_merged['PI'], predictions)
