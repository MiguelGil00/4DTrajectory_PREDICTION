import sys
import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from keras.api.optimizers import Adam

sys.path.append(os.path.join(os.getcwd(), "TFT"))

from TFM_MGARES.TFT.TFT.TFT_model import ExperimentCSVTFT, create_windows_by_callsign, GetItem, AddPositionalEncoding

# --- Configuración del experimento ----------------------------------------
model_type    = "CSV_TFT"
numeric_feat  = ["latitude", "longitude", "geoaltitude", "track", "groundspeed", "vertical_rate"]
categoric_feat= []
objective     = ["latitude", "longitude", "geoaltitude"]

csv_path      = "resultados5.csv"  # Ubicación del CSV

# Parámetros de ventana y entrenamiento
lookback      = 50
lookforward   = 1
batch_size    = 512
epochs        = 10

# Configuración del modelo TFT
model_config  = {
    "num_heads": 4,
    "ff_dim": 256,
    "num_encoder_layers": 4,
    "num_decoder_layers": 1,
    "dropout": 0.0,
    "loss": "mean_absolute_error",
    "optimizer": "adam",
    "metrics": ["mse"],
    "batch_size": batch_size
}

feat_dict = {
    "numeric": numeric_feat,
    "categoric": categoric_feat,
    "objective": objective
}

# --- Verificar CSV y datos -----------------------------------------------
try:
    df_csv = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: no se encontró el archivo {{csv_path}}.")
    sys.exit(1)

if df_csv.empty:
    print(f"Error: el CSV '{{csv_path}}' está vacío.")
    sys.exit(1)

# Instanciación del experimento con manejo de errores
try:
    experimento = ExperimentCSVTFT(
        csv_path=csv_path,
        lookback=lookback,
        lookforward=lookforward,
        model_config=model_config,
        features=feat_dict,
        batch_size=batch_size
    )
except ValueError as e:
    print("Error al procesar datos para la creación de ventanas o escalado:\n", e)
    sys.exit(1)

# --- Cargar encoder y scaler si existen -----------------------------------
encoder_path = experimento.model_path / f'encoder_{experimento.num_features}.joblib'
scaler_path  = experimento.model_path / f'scaler_{experimento.num_features}.joblib'
if encoder_path.exists() and scaler_path.exists():
    experimento.encoders          = joblib.load(encoder_path)
    experimento.scaler_objective  = joblib.load(scaler_path)
else:
    print("No se encontraron archivos de encoder/scaler. Se entrenará un modelo nuevo.")

from keras.api.models import load_model

model = load_model(
    "models/last.h5",
    custom_objects={
        "GetItem": GetItem,
        "AddPositionalEncoding": AddPositionalEncoding
    }
)

experimento.load_model()

# Recompilar el modelo para asegurar optimizador actualizado
experimento.model.compile(
    loss=model_config["loss"],
    optimizer=Adam(learning_rate=1e-3),
    metrics=model_config["metrics"]
)

experimento.model.summary()

# --- Entrenamiento opcional -----------------------------------------------
history = experimento.train(epochs=epochs)
print("Entrenamiento finalizado.")

# --- 1) Leer log de entrenamiento ----------------------------------------
progress = pd.read_csv(experimento.model_path_log)
print(progress.head())
print(progress.columns)

# --- 2) Guardar log con columna extra "model_type" ----------------------
progress["model_type"] = model_type
csv_name = f"results_{model_type}.csv"
progress.to_csv(csv_name, index=False)
print(f"Log de entrenamiento guardado en: {csv_name}")

# --- 3) Gráfica de Loss (MAE) vs Épocas ----------------------------------
skip = 0
epochs_range = range(1 + skip, progress.shape[0] + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs_range, progress["loss"].iloc[skip:], label="Training Loss")
plt.plot(epochs_range, progress["val_loss"].iloc[skip:], label="Validation Loss")
plt.title(f"{model_type} | lookback: {lookback}")
plt.xlabel("Epochs")
plt.ylabel("Loss (MAE)")
plt.legend()
plt.tight_layout()
plt.show()

# --- 4) Gráfica de MSE vs Épocas ------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, progress["mse"].iloc[skip:], label="Training MSE")
plt.plot(epochs_range, progress["val_mse"].iloc[skip:], label="Validation MSE")
plt.title(f"{model_type} | lookback: {lookback} - MSE")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 6) EVALUACIÓN ADICIONAL: RMSE y MAE en las predicciones
# ---------------------------------------------------------------------------
test_dataset = experimento._load_data("test", randomize=False).batch(batch_size)
X_test, y_test = next(iter(test_dataset))

print("X_test.shape:", X_test.shape)  # (batch_size, lookback, num_features)
print("y_test.shape:", y_test.shape)  # (batch_size, lookforward, len(objective_features))

predictions = experimento.model.predict(X_test, batch_size=batch_size)
print("Predictions.shape before reshape:", predictions.shape)

predictions = predictions.reshape(-1, len(experimento.objective_features))
print("Predictions.shape after reshape:", predictions.shape)

# Invertir la escala para volver a los valores originales
predictions_unscaled = experimento.scaler_objective.inverse_transform(predictions)
y_test_np = y_test.numpy()
y_test_unscaled = experimento.scaler_objective.inverse_transform(
    y_test_np.reshape(-1, len(experimento.objective_features))
)

rmse = np.sqrt(np.mean((predictions_unscaled - y_test_unscaled)**2, axis=0))
mae = np.mean(np.abs(predictions_unscaled - y_test_unscaled), axis=0)
print("RMSE por feature:", dict(zip(experimento.objective_features, rmse)))
print("MAE por feature:", dict(zip(experimento.objective_features, mae)))

# 7) GRÁFICA DE LATITUD PREDICHA VS REAL PARA UN CALLSIGN ESPECÍFICO
# ---------------------------------------------------------------------------
# 1) Configuración
callsign_to_plot = "3e19fd"  # ← Reemplaza con el callsign que quieras
data_callsign = experimento.df[experimento.df["callsign"] == callsign_to_plot].reset_index(drop=True)

if data_callsign.empty:
    print(f"No se encontraron datos para el callsign {callsign_to_plot}")
else:
    # 2) Crear ventanas para ese callsign
    X_callsign, y_callsign = create_windows_by_callsign(
        data_callsign,
        experimento.lookback,
        experimento.lookforward,
        experimento.numeric_features,
        experimento.categoric_features,
        experimento.objective_features,
        callsign_column="callsign"
    )

    # 3) Hacer las predicciones
    dataset_callsign = tf.data.Dataset.from_tensor_slices(X_callsign).batch(batch_size)
    preds = experimento.model.predict(dataset_callsign, verbose=0)

    # 4) Desescalar
    n_obj = len(experimento.objective_features)
    preds2d = preds.reshape(-1, n_obj)
    y2d     = y_callsign.reshape(-1, n_obj)
    preds_unsc = experimento.scaler_objective.inverse_transform(preds2d)
    y_unsc     = experimento.scaler_objective.inverse_transform(y2d)

    # 5) Extraer variables
    idx_lat = experimento.objective_features.index("latitude")
    idx_lon = experimento.objective_features.index("longitude")
    idx_geo = experimento.objective_features.index("geoaltitude")

    lat_pred = preds_unsc[:, idx_lat]
    lat_real = y_unsc[:, idx_lat]
    lon_pred = preds_unsc[:, idx_lon]
    lon_real = y_unsc[:, idx_lon]
    geo_pred = preds_unsc[:, idx_geo]
    geo_real = y_unsc[:, idx_geo]

    # 6) Construir timestamps alineados con cada predicción
    timestamps = []
    for i in range(len(y_callsign)):
        for j in range(experimento.lookforward):
            idx = i + experimento.lookback + j
            if idx < len(data_callsign):
                timestamps.append(pd.to_datetime(data_callsign["timestamp"].iloc[idx]))
            else:
                timestamps.append(pd.to_datetime(data_callsign["timestamp"].iloc[-1]))
    timestamps = pd.Series(timestamps[:len(lat_real)])

    # Asegurar que lat_pred, lat_real, timestamps tienen la misma longitud
    lat_real = lat_real[:len(timestamps)]
    lat_pred = lat_pred[:len(timestamps)]
    lon_real = lon_real[:len(timestamps)]
    lon_pred = lon_pred[:len(timestamps)]
    geo_real = geo_real[:len(timestamps)]
    geo_pred = geo_pred[:len(timestamps)]
    # print(timestamps.iloc[0:100])

    # 7a) Gráfica LATITUD (30 segundos)
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, lat_real, label="Latitud real")
    plt.plot(timestamps, lat_pred, label="Latitud predicha", marker="o")
    plt.title(f"Latitud real vs predicha para {callsign_to_plot} (30 segundos)")
    plt.xlabel("Timestamp")
    plt.ylabel("Latitud (°)")
    plt.xticks(rotation=45)
    # plt.ylim([50.12, 50.16])
    plt.xlim([timestamps.iloc[225], timestamps.iloc[225] + pd.Timedelta(seconds=30)])
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 7b) Gráfica LONGITUD (1 minuto)
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, lon_real, label="Longitud real")
    plt.plot(timestamps, lon_pred, label="Longitud predicha", marker="o")
    plt.title(f"Longitud real vs predicha para {callsign_to_plot} (30 segundos)")
    plt.xlabel("Timestamp")
    plt.ylabel("Longitud (°)")
    plt.xticks(rotation=45)
    # plt.ylim([8.325, 8.36])
    plt.xlim([timestamps.iloc[225], timestamps.iloc[225] + pd.Timedelta(seconds=30)])
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 7c) Gráfica ALTITUD 
    df_geo = pd.DataFrame({
        "timestamp": timestamps,
        "geo_real": geo_real,
        "geo_pred": geo_pred
    })

    # Índice temporal para resample
    df_geo.set_index("timestamp", inplace=True)

    # Submuestreo cada 5 segundos con media
    df_geo_resampled = df_geo.resample("5S").mean().dropna().reset_index()

    # Columnas finales para graficar
    ts_geo     = df_geo_resampled["timestamp"]
    geo_real_s = df_geo_resampled["geo_real"]
    geo_pred_s = df_geo_resampled["geo_pred"]

    # Gráfica
    plt.figure(figsize=(12, 6))
    plt.plot(ts_geo, geo_real_s, label="Altitud real")
    plt.plot(ts_geo, geo_pred_s, label="Altitud predicha", marker="o")
    plt.title(f"Altitud real vs predicha para {callsign_to_plot} (muestreo cada 5 segundos)")
    plt.xlabel("Timestamp")
    plt.ylabel("Geoaltitude (m)")
    plt.xticks(rotation=45)
    plt.xlim([timestamps.iloc[0], timestamps.iloc[0] + pd.Timedelta(minutes=12)])
    plt.legend()
    plt.tight_layout()
    plt.show()

# 8) Guardar resultados en CSV
df_resultados = pd.DataFrame({
    "timestamp": timestamps,
    "lat_real": lat_real,
    "lat_pred": lat_pred,
    "lon_real": lon_real,
    "lon_pred": lon_pred,
    "geoalt_real": geo_real,
    "geoalt_pred": geo_pred
})

# Guardar en archivo CSV
output_path = f"resultados_{callsign_to_plot}.csv"
df_resultados.to_csv(output_path, index=False)
print(f"Resultados guardados en: {output_path}")

# 7d) Cálculo y representación del % de error relativo por timestamp
import matplotlib.dates as mdates

# Asegurar misma longitud
preds_unsc = preds_unsc[:len(y_unsc)]
y_unsc     = y_unsc[:len(preds_unsc)]
timestamps = timestamps[:len(y_unsc)]

# Nombre de variables
features = experimento.objective_features
colores = ['tab:blue', 'tab:orange', 'tab:green']

# Crear DataFrame general
df_all = pd.DataFrame(preds_unsc, columns=[f"{f}_pred" for f in features])
for i, f in enumerate(features):
    df_all[f"{f}_real"] = y_unsc[:, i]
df_all["timestamp"] = timestamps

# Submuestreo: cada 10 pasos
df_all = df_all.iloc[::10].reset_index(drop=True)

# Plot del % de error por feature
for i, feature in enumerate(features):
    pred_col = f"{feature}_pred"
    real_col = f"{feature}_real"

    # Cálculo del error relativo en porcentaje
    df_all[f"{feature}_perc_error"] = 100 * np.abs(df_all[pred_col] - df_all[real_col]) / df_all[real_col].replace(0, np.nan)

    # Plot
    plt.figure(figsize=(14, 4))
    plt.plot(df_all["timestamp"], df_all[f"{feature}_perc_error"], color=colores[i],
             label=f"% error relativo - {feature}", marker="o", linestyle="-")
    plt.title(f"% de error relativo en {feature} a lo largo del tiempo (cada 10 segundos)")
    plt.xlabel("Timestamp")
    plt.ylabel("Error (%)")
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
