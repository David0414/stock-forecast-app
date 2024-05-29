import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Generar datos de ejemplo (o cargar datos reales)
# Aquí, creamos un ejemplo simple de datos de una serie temporal
# Puedes reemplazar esto con datos reales de tu serie temporal
data = {
    'ds': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'y': [i + (i * 0.1) for i in range(100)]
}
df = pd.DataFrame(data)

# Visualizar los datos originales
plt.figure(figsize=(10, 6))
plt.plot(df['ds'], df['y'], label='Original Data')
plt.title('Original Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Preparar el modelo Prophet
m = Prophet()

# Ajustar el modelo a los datos históricos
m.fit(df)

# Crear un dataframe de fechas futuras para predicción
future = m.make_future_dataframe(periods=30)

# Hacer predicciones
forecast = m.predict(future)

# Visualizar las predicciones
fig1 = m.plot(forecast)
plt.title('Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Visualizar los componentes de la predicción
fig2 = m.plot_components(forecast)
plt.show()

# Mostrar las últimas filas del dataframe de predicciones
print(forecast.tail())
