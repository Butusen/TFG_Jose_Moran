import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
from sklearn.metrics import r2_score

#warnings.filterwarnings("ignore")

df_train = pd.read_csv("sales_train.csv")
df_shops = pd.read_csv("shops-translated.csv")
df_items = pd.read_csv("item_category.csv")
df_calendar = pd.read_csv("calendar.csv")


df_train['date'] = pd.to_datetime(df_train['date'], format="%d.%m.%Y")
df_calendar['date'] = pd.to_datetime(df_calendar['date'])

df_nuevo = pd.merge(df_train, df_shops, on='shop_id', how="left")
df_final = pd.merge(df_nuevo, df_items, on="item_id", how="left")


df_bo2 = df_final[df_final['item_id']== 2308]
#df_bo2 = df_bo2[ df_final['City'] == "Moscow"]

df_bo2_transformado = df_bo2[['date', 'item_cnt_day']]
df_bo2_transformado['date'] = pd.to_datetime(df_bo2_transformado['date'])
weeks=df_bo2_transformado.groupby([pd.Grouper(key='date', freq='W')])['item_cnt_day'].sum()
df_weeks_bo2 = pd.DataFrame(weeks)
df_weeks_bo2.plot()
df_bo2_train = df_weeks_bo2.iloc[0:104]
df_bo2_test = df_weeks_bo2.iloc[104:148]
df_bo2_train.plot()
df_bo2_test.plot()
log_bo2 = np.log(df_bo2_train)#.plot()
#Con el log comprobamos si es estable en varianza, no nos sirve
estacionaria_bo2 = df_bo2_train.diff()
estacionaria_bo2_test = df_bo2_test.diff()
estacionaria_bo2.plot(color="red")
estacionaria_bo2_test.plot(color ="blue")
#Tiene una tenedencia repetida anualmente
acf = plot_acf(df_bo2_train)
#pacf = plot_pacf(df_bo2_train)
dftest = adfuller(df_bo2_train, autolag='AIC')
## Es estacionaria porque el p-valor < 0.05, es 2.88 x 10^-6
hola = auto_arima(df_bo2_train, trace=True, suppress_warnings=True)
hola.summary()

def checkings(df,df2,p,d,q,p2,d2,q2):
    fit_arima = ARIMA(df, order=(p,d,q),seasonal_order=(p2, d2, q2, 52))
    fit_arima2 = fit_arima.fit()
    fit_arima2.summary()
    predict = fit_arima2.predict(start=105,end=147).to_frame()
    comparacion = predict.join(df2)
    plt.plot(comparacion)
    plt.show()
    mse = mean_squared_error(comparacion['item_cnt_day'], comparacion['predicted_mean'])
    r_2 = r2_score(comparacion['item_cnt_day'], comparacion['predicted_mean'])
    print(fit_arima2.summary())
    print("MSE="+str(mse))
    print("r^2="+str(r_2))
    
checkings(df_bo2_train,df_bo2_test,1,0,0,1,0,0)
checkings(df_bo2_train,df_bo2_test,1,0,0,1,1,0)
checkings(df_bo2_train,df_bo2_test,1,0,0,1,0,1)
checkings(df_bo2_train,df_bo2_test,1,0,0,1,1,1)
checkings(df_bo2_train,df_bo2_test,1,0,0,0,1,0)
checkings(df_bo2_train,df_bo2_test,1,0,0,2,0,0)
checkings(df_bo2_train,df_bo2_test,1,1,0,1,0,0)
checkings(df_bo2_train,df_bo2_test,1,0,1,1,0,0)
checkings(df_bo2_train,df_bo2_test,1,0,1,1,1,0)
checkings(df_bo2_train,df_bo2_test,1,0,1,1,1,1)
checkings(df_bo2_train,df_bo2_test,2,1,0,1,0,0)
checkings(df_bo2_train,df_bo2_test,2,1,1,1,0,0)
checkings(estacionaria_bo2,estacionaria_bo2_test,1,1,0,1,0,0)
checkings(df_bo2_train,df_bo2_test,0,2,2,1,0,0)

#Ahora probamos con todos los datos
prueba_total = auto_arima(df_weeks_bo2, trace=True, suppress_warnings=True)
prueba_total.summary()

weekly_dates = pd.date_range(start="2015-11-08", end="2016-12-31", freq='W')
weekly_df = pd.DataFrame(index=weekly_dates)
df_cuatro_años= pd.concat([df_weeks_bo2,weekly_df])

fit_arima_real = ARIMA(df_weeks_bo2, order=(1,1,0),seasonal_order=(1, 0, 0, 52))
fit_arima2_real = fit_arima_real.fit()
fit_arima2_real.summary()
predict = fit_arima2_real.predict(start=147,end=207).to_frame()
plt.plot(df_weeks_bo2)
plt.plot(predict)
plt.xticks(rotation='vertical')
plt.show()

