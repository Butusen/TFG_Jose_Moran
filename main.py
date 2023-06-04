import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


df_train = pd.read_csv("sales_train.csv")
df_shops = pd.read_csv("shops-translated.csv")
df_items = pd.read_csv("item_category.csv")
df_calendar = pd.read_csv("calendar.csv")
df_censo = pd.read_csv("censoRusia.csv",sep=";")
df_shops = pd.merge(df_shops,df_censo, on="City", how="left")
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
plt.title("Unidades vendidas de Call of Duty:Black Ops 2(PC Jewel)")
plt.xlabel("Fecha")
plt.ylabel("Cantidad")
plt.show()
df_bo2_train = df_weeks_bo2.iloc[0:104]
df_bo2_test = df_weeks_bo2.iloc[104:148]

log_bo2 = np.log(df_bo2_train).plot()
plt.title("Datos de entrenamiento aplicando logaritmo")
plt.xlabel("Fecha")
plt.show()
#Con el log comprobamos si es estable en varianza, no nos sirve

result = seasonal_decompose(df_weeks_bo2['item_cnt_day'], model='multiplicative')

result.seasonal.plot()
plt.title("Gráfico de estacionalidad")
plt.show()

estacionaria_bo2 = df_bo2_train.diff()
estacionaria_bo2_test = df_bo2_test.diff()
plt.plot(estacionaria_bo2,label="train",color="red")
plt.plot(estacionaria_bo2_test,label="test",color ="blue")
plt.xticks(rotation='vertical')
plt.title("Gráfico de entrenamiento aplicando diferenciación")
plt.xlabel("Fecha")
plt.ylabel("Diferencia")
plt.legend()
plt.show()
#Tiene una tenedencia repetida anualmente
acf = plot_acf(df_bo2_train)
pacf = plot_pacf(df_bo2_train)
dftest = adfuller(df_bo2_train, autolag='AIC')
## Es estacionaria porque el p-valor < 0.05, es 2.88 x 10^-6

def checkings(df,df2,p,d,q,p2,d2,q2):
    fit_arima = ARIMA(df, order=(p,d,q),seasonal_order=(p2, d2, q2, 52))
    fit_arima2 = fit_arima.fit()
    fit_arima2.summary()
    predict = fit_arima2.predict(start=105,end=147).to_frame()
    comparacion = predict.join(df2)
    ax = comparacion.plot(y=['item_cnt_day', 'predicted_mean'])
    ax.legend(['datos reales', 'datos pronosticados'])
    ax.set_title(f"Comparación datos reales y pronosticados 2015\norder(p={p}, d={d}, q={q}) seasonal(P={p2}, D={d2}, Q={q2}, 52)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Nº de unidades")
    plt.show()
    mse = mean_squared_error(comparacion['item_cnt_day'], comparacion['predicted_mean'])
    r_2 = r2_score(comparacion['item_cnt_day'], comparacion['predicted_mean'])
    print(fit_arima2.summary())
    print("MSE="+str(mse))
    
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


#Ahora probamos con todos los datos

fit_arima_real = ARIMA(df_weeks_bo2, order=(1,1,0),seasonal_order=(1, 0, 0, 52))
fit_arima2_real = fit_arima_real.fit()
fit_arima2_real.summary()
predict = fit_arima2_real.predict(start=147,end=207).to_frame()
plt.plot(df_weeks_bo2,label="datos reales")
plt.title("Ventas reales y pronosticadas \n de Call of Duty:Black Ops 2(PC Jewel) entre 2013-2016")
plt.plot(predict, label="datos pronosticados")
plt.xticks(rotation='vertical')
plt.xlabel("Fecha")
plt.ylabel("Nº de unidades vendidas")
plt.legend()
plt.show()


