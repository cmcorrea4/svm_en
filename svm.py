import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def load_and_process_data(file):
    """Carga y procesa el archivo CSV"""
    df = pd.read_csv(file)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    # Extraer características temporales
    df['Hour'] = df['Datetime'].dt.hour
    df['Minute'] = df['Datetime'].dt.minute
    return df

def create_features(df):
    """Crea características para el modelo"""
    X = df[['Hour', 'Minute', 'Kwh']].values
    return X

def train_model(X, threshold):
    """Entrena el modelo SVM"""
    # Crear etiquetas basadas en el umbral
    y = (X[:, 2] <= threshold).astype(int)  # 1 para normal, 0 para anómalo
    
    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Entrenar modelo
    svm = SVC(kernel='rbf')
    svm.fit(X_scaled, y)
    
    return svm, scaler

def main():
    st.title("Detector de Anomalías en Consumo Energético")
    
    # Sidebar para configuración
    st.sidebar.header("Configuración")
    
    # Carga de archivo de entrenamiento
    st.sidebar.subheader("Archivo de Entrenamiento")
    training_file = st.sidebar.file_uploader("Cargar archivo CSV de entrenamiento", type=['csv'])
    
    if training_file is not None:
        # Cargar y mostrar datos de entrenamiento
        df_train = load_and_process_data(training_file)
        
        st.subheader("Datos de Entrenamiento")
        st.write(df_train)
        
        # Slider para umbral
        max_kwh = float(df_train['Kwh'].max())
        threshold = st.sidebar.slider(
            "Umbral de consumo normal (kWh)",
            min_value=0.0,
            max_value=max_kwh,
            value=max_kwh/2,
            step=0.1
        )
        
        # Crear y entrenar modelo
        X_train = create_features(df_train)
        model, scaler = train_model(X_train, threshold)
        
        # Mostrar distribución de clases en datos de entrenamiento
        normal_count = sum(df_train['Kwh'] <= threshold)
        anomaly_count = len(df_train) - normal_count
        
        st.subheader("Distribución de Clases en Entrenamiento")
        st.write(f"Consumos Normales (≤ {threshold:.2f} kWh): {normal_count}")
        st.write(f"Consumos Anómalos (> {threshold:.2f} kWh): {anomaly_count}")
        
        # Carga de archivo para predicción
        st.sidebar.subheader("Archivo para Predicción")
        prediction_file = st.sidebar.file_uploader("Cargar archivo CSV para predicción", type=['csv'])
        
        if prediction_file is not None:
            # Cargar y procesar datos para predicción
            df_pred = load_and_process_data(prediction_file)
            X_pred = create_features(df_pred)
            
            # Realizar predicciones
            X_pred_scaled = scaler.transform(X_pred)
            predictions = model.predict(X_pred_scaled)
            
            # Agregar predicciones al DataFrame
            df_pred['Clasificación'] = ['Normal' if pred == 1 else 'Anómalo' for pred in predictions]
            
            # Mostrar resultados
            st.subheader("Resultados de la Predicción")
            st.write(df_pred)
            
            # Mostrar estadísticas de predicción
            st.subheader("Estadísticas de Predicción")
            pred_normal = sum(predictions == 1)
            pred_anomaly = sum(predictions == 0)
            st.write(f"Consumos Clasificados como Normales: {pred_normal}")
            st.write(f"Consumos Clasificados como Anómalos: {pred_anomaly}")

if __name__ == "__main__":
    main()
