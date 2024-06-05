import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.express as px
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
TRAIN_END = "2023-12-31"

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    try:
        data = yf.download(ticker, START, TODAY, progress=False)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
if data is not None:
    data_load_state.text('Loading data... done!')
else:
    data_load_state.text('Failed to load data.')

if data is not None:
    st.subheader('Raw data')
    st.write(data.tail())

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.subheader('Forecast data')
    st.write(forecast.tail())

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    

    # Añadir la función para calcular el MAPE
    def calculate_mape(actual, predicted):
        return np.mean(np.abs((actual - predicted) / actual)) * 100

    # Alinear las fechas
    forecast_filtered = forecast.set_index('ds').loc[data['Date']]
    actual_filtered = data.set_index('Date').loc[forecast_filtered.index]

    # Calcular el MAPE
    mape = calculate_mape(actual_filtered['Close'], forecast_filtered['yhat'])

    # Mostrar el MAPE en la aplicación
    st.subheader('Error del pronóstico')
    st.write(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

# Apartado adicional para predicciones hasta 2023 y comparación con 2024
st.title('Projection and Comparison for 2024')

train_data = data[data['Date'] <= TRAIN_END]
df_train = train_data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=365)  # Proyectar para el año 2024
forecast = m.predict(future)

st.write('Prediction Chart for 2024')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)



# Comparar las predicciones con los datos reales de 2024
if date.today().year == 2024:
    real_2024 = data[data['Date'] > TRAIN_END]
    if not real_2024.empty:
        # Alinear las fechas
        forecast_2024 = forecast.set_index('ds').loc[real_2024['Date']]
        real_2024 = real_2024.set_index('Date').loc[forecast_2024.index]

        # Calcular el MAPE
        mape = calculate_mape(real_2024['Close'], forecast_2024['yhat'])

        # Mostrar el MAPE en la aplicación
        st.subheader('Error percentage for 2024')
        st.write(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

        # Graficar datos reales y predicciones usando barras
        comparison_df = forecast_2024.copy()
        comparison_df['Real'] = real_2024['Close']

        fig = go.Figure(data=[
            go.Bar(name='Real Data', x=comparison_df.index, y=comparison_df['Real'], marker_color='blue'),
            go.Bar(name='Predicted Data', x=comparison_df.index, y=comparison_df['yhat'], marker_color='red')
        ])

        fig.update_layout(barmode='group', title='Real vs Predicted Data for 2024', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)

        # Gráfico de círculo para mostrar el porcentaje de error
        labels = ['Accurate', 'Error']
        values = [100 - mape, mape]
        pie_chart = px.pie(values=values, names=labels, title='Error Percentage for 2024', color_discrete_sequence=['green', 'red'])
        st.plotly_chart(pie_chart)
    else:
        st.subheader('No hay datos reales para 2024 para comparar.')
else:
    st.subheader('Esperando datos de 2024 para comparar.')
