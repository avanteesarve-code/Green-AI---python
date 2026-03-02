import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(page_title="Energy Consumption Predictor", layout="wide")

st.title("🌍 Climate Based Energy Consumption Prediction")
st.write("Upload climate_data.csv to train the model and predict energy consumption.")

# Upload CSV
uploaded_file = st.file_uploader("Upload climate_data.csv", type=["csv"])

if uploaded_file is not None:
    
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    if "Energy Consumption" not in data.columns:
        st.error("Column 'Energy Consumption' not found in dataset!")
    else:
        X = data.drop(columns=["Energy Consumption"])
        y = data["Energy Consumption"]

        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Model
        model = Sequential()
        model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(1, activation='linear'))

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )

        st.write("Training model... ⏳")

        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )

        st.success("Model Training Completed ✅")

        # Evaluation
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

        st.subheader("Model Performance")
        st.write(f"Test Loss (MSE): {test_loss:.4f}")
        st.write(f"Test MAE: {test_mae:.4f}")

        # Plot Loss
        st.subheader("Training vs Validation Loss")

        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Train Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        st.pyplot(fig)

        # Predictions
        predictions = model.predict(X_test)

        st.subheader("Sample Predictions")
        results = pd.DataFrame({
            "Actual": y_test.values[:5],
            "Predicted": predictions.flatten()[:5]
        })

        st.dataframe(results)

        # Manual Prediction
        st.subheader("🔮 Predict New Data")

        input_data = []
        for column in X.columns:
            value = st.number_input(f"Enter {column}", value=float(data[column].mean()))
            input_data.append(value)

        if st.button("Predict Energy Consumption"):
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)
            st.success(f"Predicted Energy Consumption: {prediction[0][0]:.2f}")