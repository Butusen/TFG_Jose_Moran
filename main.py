import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

df_train = pd.read_csv("sales_train.csv")
df_shops = pd.read_csv("shops-translated.csv")
df_items = pd.read_csv("item_category.csv")
df_calendar = pd.read_csv("calendar.csv")


df_train['date'] = pd.to_datetime(df_train['date'], format="%d.%m.%Y")
df_calendar['date'] = pd.to_datetime(df_calendar['date'])

df_nuevo = pd.merge(df_train, df_shops, on='shop_id', how="left")
df_final = pd.merge(df_nuevo, df_items, on="item_id", how="left")


df_bo2 = df_final[df_final['item_id']== 2308]

df_bo2_transformado = df_bo2[['date', 'item_cnt_day']]
df_bo2_transformado['date'] = pd.to_datetime(df_bo2_transformado['date'])
weeks=df_bo2_transformado.groupby([pd.Grouper(key='date', freq='W')])['item_cnt_day'].sum()
df_weeks = pd.DataFrame(weeks)
#agregar = {'date':'first','item_cnt_day':'sum'}
#df_bo2_agregado = df_bo2_transformado.groupby('date').item_cnt_day.agg(['sum'])
#df_bo2_agregado['date'] = df_bo2_agregado.index

dftest = adfuller(df_weeks, autolag='AIC')

hola = auto_arima(df_weeks['item_cnt_day'], trace=True, suppress_warnings=True)
hola.summary()

df_weeks['item_cnt_day'].plot(legend=True)
df_weeks['ln'] = np.log(df_weeks['item_cnt_day'])
df_weeks['estacionaria'] = df_weeks['item_cnt_day'].diff()
df_weeks['ln'].plot(legend=True)
df_weeks['estacionaria'].plot(legend=True)
# Cambia, no es muy estable en varianza
df_bo2_train = df_weeks.iloc[0:130]
df_bo2_test = df_weeks.iloc[130:143]

model = ARIMA(df_bo2_train['item_cnt_day'], order=(1,1,0))
model = model.fit()
model.summary()

inicio= len(df_bo2_train)-1
fin = len(df_bo2_train) + len(df_bo2_test)-2
prediccion = model.predict(start=inicio, end=fin, typ='levels')
prediccion.index = df_weeks.index[inicio:fin+1]

prediccion.plot(legend=True)
df_bo2_test['item_cnt_day'].plot(legend=True)

df_bo2_test['item_cnt_day'].mean()
rmse = sqrt(mean_squared_error(prediccion, df_bo2_test['sum']))




