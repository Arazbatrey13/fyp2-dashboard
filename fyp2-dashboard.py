import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ✅ Safe TensorFlow import
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
except ImportError:
    st.error("❌ TensorFlow is not installed. Please add 'tensorflow-cpu==2.11.0' in requirements.txt.")
    st.stop()

st.set_page_config(page_title="Stock Forecasting", layout="wide")
st.title("📈 Stock Forecasting Web App")

# Sidebar controls
stock_choice = st.sidebar.selectbox("Choose Stock", ["AAPL", "SAP.DE", "005930.KS"],
    format_func=lambda x: {"AAPL":"Apple", "SAP.DE":"SAP", "005930.KS":"Samsung"}[x])
model_choice = st.sidebar.selectbox("Choose Model", ["SVR", "KNN", "Random Forest", "XGBoost", "CNN+LSTM"])
seq_len = st.sidebar.slider("Sequence Length (CNN+LSTM)", 30, 100, 60, 10)
epochs = st.sidebar.slider("Epochs (CNN+LSTM)", 5, 50, 10, 5)

# Load data
st.write(f"Fetching **{stock_choice}** data...")
df_raw = yf.download(stock_choice, start='2010-01-01', end='2020-12-31')

if isinstance(df_raw.columns, pd.MultiIndex):
    try:
        df = df_raw[stock_choice][['Close']].copy()
    except KeyError:
        st.error(f"Could not find 'Close' column for {stock_choice}.")
        st.stop()
else:
    try:
        df = df_raw[['Close']].copy()
    except KeyError:
        st.error("The 'Close' column is not available in the downloaded data.")
        st.dataframe(df_raw.head())
        st.stop()

df = df.dropna()
st.line_chart(df)

# Classical features
df['Target'] = df['Close'].shift(-1)
df['MA5'] = df['Close'].rolling(5).mean()
df = df.dropna()
X_cls = df[['Close', 'MA5']]
y_cls = df['Target']

split = int(0.8 * len(X_cls))
X_train_cls, X_test_cls = X_cls.iloc[:split], X_cls.iloc[split:]
y_train_cls, y_test_cls = y_cls.iloc[:split], y_cls.iloc[split:]

scaler_cls = StandardScaler()
X_train_scaled = scaler_cls.fit_transform(X_train_cls)
X_test_scaled = scaler_cls.transform(X_test_cls)

def directional_acc(y_true, y_pred, baseline):
    return np.mean(np.where(y_true > baseline, 1, 0) == np.where(y_pred > baseline, 1, 0)) * 100

# CNN+LSTM data
scaler_lstm = MinMaxScaler()
scaled_data = scaler_lstm.fit_transform(df[['Close']])
X_lstm, y_lstm = [], []
for i in range(seq_len, len(scaled_data)):
    X_lstm.append(scaled_data[i-seq_len:i, 0])
    y_lstm.append(scaled_data[i, 0])
X_lstm = np.array(X_lstm).reshape(-1, seq_len, 1)
y_lstm = np.array(y_lstm)

split_lstm = int(0.8 * len(X_lstm))
X_train_lstm, X_test_lstm = X_lstm[:split_lstm], X_lstm[split_lstm:]
y_train_lstm, y_test_lstm = y_lstm[:split_lstm], y_lstm[split_lstm:]

# Train & predict
if model_choice == "SVR":
    model = SVR(kernel='rbf')
    model.fit(X_train_scaled, y_train_cls)
    pred = model.predict(X_test_scaled)
    baseline = X_test_cls['Close'].values
elif model_choice == "KNN":
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train_scaled, y_train_cls)
    pred = model.predict(X_test_scaled)
    baseline = X_test_cls['Close'].values
elif model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train_cls)
    pred = model.predict(X_test_scaled)
    baseline = X_test_cls['Close'].values
elif model_choice == "XGBoost":
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train_cls)
    pred = model.predict(X_test_scaled)
    baseline = X_test_cls['Close'].values
else:
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(seq_len,1)),
        MaxPooling1D(2),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train_lstm, y_train_lstm, epochs=epochs, batch_size=32, verbose=0)
    pred_scaled = model.predict(X_test_lstm)
    pred = scaler_lstm.inverse_transform(pred_scaled)
    actual = scaler_lstm.inverse_transform(y_test_lstm.reshape(-1,1))
    baseline = scaler_lstm.inverse_transform(X_test_lstm[:,-1,:])
    mse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    diracc = directional_acc(actual.flatten(), pred.flatten(), baseline.flatten())
    st.subheader(f"Model Performance: {model_choice}")
    st.metric("MSE", f"{mse:.4f}")
    st.metric("MAE", f"{mae:.4f}")
    st.metric("MAPE (%)", f"{mape:.2f}")
    st.metric("Directional Accuracy (%)", f"{diracc:.2f}")
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(actual, label='Actual', color='blue')
    ax.plot(pred, label='Predicted', color='orange')
    ax.legend()
    st.pyplot(fig)
    st.stop()

# Classical model metrics
mse = mean_squared_error(y_test_cls, pred)
mae = mean_absolute_error(y_test_cls, pred)
mape = np.mean(np.abs((y_test_cls - pred) / y_test_cls)) * 100
diracc = directional_acc(y_test_cls.values, pred, baseline)

st.subheader(f"Model Performance: {model_choice}")
st.metric("MSE", f"{mse:.4f}")
st.metric("MAE", f"{mae:.4f}")
st.metric("MAPE (%)", f"{mape:.2f}")
st.metric("Directional Accuracy (%)", f"{diracc:.2f}")

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(y_test_cls.values, label='Actual', color='blue')
ax.plot(pred, label='Predicted', color='orange')
ax.legend()
st.pyplot(fig)
