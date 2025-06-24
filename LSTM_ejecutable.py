import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import joblib
from keras.api.optimizers import Adam
from matplotlib.ticker import MultipleLocator

sys.path.append(os.path.join(os.getcwd(), "rtaUtils"))
from rtaUtils.LSTM_ejecutable import ExperimentCSVVanilla, create_windows_by_callsign # Asegúrate de que el paquete tenga __init__.py

### Configuración del experimento #############################################
model_type   = "CSV_LSTM"
# Definir las features para el CSV
numeric_feat   = ["latitude", "longitude", "geoaltitude", "track", "groundspeed", "vertical_rate"]
categoric_feat = []
objective      = ["latitude", "longitude", "geoaltitude"]

csv_path     = "resultados5.csv"  # Ubicación del CSV

# Configuración del modelo
lookback     = 50
lookforward  = 1
n_units      = 20  # Número de neuronas
act_function = "relu"  # <--- cámbialo a "sigmoid", "linear", etc. en cada ejecución
batch_size   = 512
epochs       = 10

model_config = {
    "n_units": n_units,
    "act_function": act_function,
    "batch_size": batch_size
}

feat_dict = {
    "numeric": ["latitude", "longitude", "geoaltitude", "track", "groundspeed", "vertical_rate"],
    "categoric": [],
    "objective": ["latitude", "longitude", "geoaltitude"]
}

### Instanciación del experimento #############################################
experimento = ExperimentCSVVanilla(
    csv_path=csv_path,
    lookback=lookback,
    lookforward=lookforward,
    model_config=model_config,
    features=feat_dict,
    batch_size=batch_size
)

encoder_path = experimento.model_path / f'encoder_{experimento.num_features}.joblib'
scaler_path = experimento.model_path / f'scaler_{experimento.num_features}.joblib'
if encoder_path.exists() and scaler_path.exists():
    experimento.encoders = joblib.load(encoder_path)
    experimento.scaler = joblib.load(scaler_path)
else:
    print("No se encontraron archivos de encoder/scaler. Se entrenará un modelo nuevo.")

# Si deseas entrenar SIEMPRE desde cero, comenta la siguiente línea
experimento.load_model()

# Recompilar el modelo con un nuevo optimizador para que se vincule a las variables actuales:
experimento.model.compile(
    loss='mean_absolute_error',
    optimizer=Adam(learning_rate=1e-3),
    metrics=['mean_squared_error']
)

experimento.model.summary()

### Entrenamiento #############################################################
# history = experimento.train(epochs=epochs)
# print("Entrenamiento finalizado.")

# ---------------------------------------------------------------------------
# 1) LEER EL LOG DE ENTRENAMIENTO
# ---------------------------------------------------------------------------
progress = pd.read_csv(experimento.model_path_log)
print(progress.head())
print(progress.columns)

# ---------------------------------------------------------------------------
# 2) GUARDAR EL LOG CON UNA COLUMNA EXTRA "activation"
#    Para luego poder juntar varios logs (por ejemplo, "tanh", "sigmoid", etc.)
# ---------------------------------------------------------------------------
progress["activation"] = act_function
csv_name = f"results_{act_function}.csv"
progress.to_csv(csv_name, index=False)
print(f"Log de entrenamiento guardado en: {csv_name}")

# ---------------------------------------------------------------------------
# 3) GRÁFICA DE LOSS (MAE) vs ÉPOCAS
# ---------------------------------------------------------------------------
skip = 0
epochs_range = range(1 + skip, progress.shape[0] + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs_range, progress["loss"].iloc[skip:], label="Training Loss")
plt.plot(epochs_range, progress["val_loss"].iloc[skip:], label="Validation Loss")
plt.title(f"{model_type} | n_units: {n_units} | lookback: {lookback}")
plt.xlabel("Epochs")
plt.ylabel("Loss (MAE)")
plt.legend()
plt.tight_layout()
# plt.show()

# ---------------------------------------------------------------------------
# 4) GRÁFICA DE MSE vs ÉPOCAS
# ---------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, progress["mean_squared_error"].iloc[skip:], label="Training MSE")
plt.plot(epochs_range, progress["val_mean_squared_error"].iloc[skip:], label="Validation MSE")
plt.title(f"{model_type} | n_units: {n_units} | lookback: {lookback} - MSE")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
# plt.show()

# ---------------------------------------------------------------------------
# 5) GRÁFICA SOLO DE VALIDACIÓN CON EJE CON INCREMENTOS PEQUEÑOS Y SIN SUAVIZADO
# ---------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, progress["val_mean_squared_error"].iloc[skip:], 
         marker='o', linestyle='-', drawstyle='default', label="Validation MSE")
plt.xlabel("Epochs")
plt.ylabel("Validation MSE")
plt.title("Validation MSE vs. Epochs")
plt.legend()

# Configuramos el eje Y para que tenga saltos de 0.001
ax = plt.gca()
ax.set_ylim([0, 0.005])  # O lo que consideres adecuado
ax.set_xlim([10, 60])    # Ajuste de rango en eje X: solo de época 5 a 10

plt.tight_layout()
# plt.show()

# ---------------------------------------------------------------------------
# 6) EVALUACIÓN ADICIONAL: RMSE y MAE en las predicciones
# ---------------------------------------------------------------------------
# test_dataset = experimento._load_data("test", randomize=False).batch(batch_size)
# X_test, y_test = next(iter(test_dataset))

# print("X_test.shape:", X_test.shape)  # (batch_size, lookback, num_features)
# print("y_test.shape:", y_test.shape)  # (batch_size, lookforward, len(objective_features))

# predictions = experimento.model.predict(X_test, batch_size=batch_size)
# print("Predictions.shape before reshape:", predictions.shape)

# predictions = predictions.reshape(-1, len(experimento.objective_features))
# print("Predictions.shape after reshape:", predictions.shape)

# # Invertir la escala para volver a los valores originales
# predictions_unscaled = experimento.scaler_objective.inverse_transform(predictions)
# y_test_np = y_test.numpy()
# y_test_unscaled = experimento.scaler_objective.inverse_transform(
#     y_test_np.reshape(-1, len(experimento.objective_features))
# )

# y_true_all = []
# y_pred_all = []

# for X_batch, y_batch in test_dataset:
#     y_pred = experimento.model.predict(X_batch, batch_size=batch_size)
#     y_pred = y_pred.reshape(-1, len(experimento.objective_features))
#     y_batch = y_batch.numpy().reshape(-1, len(experimento.objective_features))

#     y_pred_all.append(y_pred)
#     y_true_all.append(y_batch)

# # Unir todos los batches
# y_pred_all = np.vstack(y_pred_all)
# y_true_all = np.vstack(y_true_all)

# # Invertir escala
# y_pred_unscaled = experimento.scaler_objective.inverse_transform(y_pred_all)
# y_true_unscaled = experimento.scaler_objective.inverse_transform(y_true_all)

# # Calcular errores globales
# rmse = np.sqrt(np.mean((y_pred_unscaled - y_true_unscaled) ** 2, axis=0))
# mae = np.mean(np.abs(y_pred_unscaled - y_true_unscaled), axis=0)

# denominador = np.where(y_true_unscaled == 0, np.nan, y_true_unscaled)
# percent_error = np.nanmean(np.abs((y_true_unscaled - y_pred_unscaled) / denominador), axis=0) * 100

# # --- NUEVO: %Error total en 3D ---
# # Distancia real y predicha
# distancias = np.linalg.norm(y_true_unscaled - y_pred_unscaled, axis=1)
# magnitud_real = np.linalg.norm(y_true_unscaled, axis=1)
# magnitud_real_safe = np.where(magnitud_real == 0, np.nan, magnitud_real)

# percent_error_3d = np.nanmean(distancias / magnitud_real_safe) * 100

# # Mostrar resultados
# print("\n--- Evaluación del modelo ---")
# print("MAE por característica:", dict(zip(experimento.objective_features, mae)))
# print("RMSE por característica:", dict(zip(experimento.objective_features, rmse)))
# print("%Error por característica:", dict(zip(experimento.objective_features, percent_error)))
# print("%Error total 3D:", percent_error_3d)


# 7) GRÁFICA DE LATITUD PREDICHA VS REAL PARA UN CALLSIGN ESPECÍFICO
# ---------------------------------------------------------------------------
# 1) Configuración
callsign_to_plot = "3c64ee"  # ← Reemplaza con el callsign que quieras
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

    # # 7a) Gráfica LATITUD (30 segundos)
    # plt.figure(figsize=(12, 6))
    # plt.plot(timestamps, lat_real, label="Latitud real")
    # plt.plot(timestamps, lat_pred, label="Latitud predicha", marker="o")
    # plt.title(f"Latitud real vs predicha para {callsign_to_plot} (30 segundos)")
    # plt.xlabel("Timestamp")
    # plt.ylabel("Latitud (°)")
    # plt.xticks(rotation=45)
    # plt.ylim([50.12, 50.16])
    # plt.xlim([timestamps.iloc[225], timestamps.iloc[225] + pd.Timedelta(seconds=30)])
    # plt.legend()
    # plt.tight_layout()
    # # plt.show()

    # # 7b) Gráfica LONGITUD (1 minuto)
    # plt.figure(figsize=(12, 6))
    # plt.plot(timestamps, lon_real, label="Longitud real")
    # plt.plot(timestamps, lon_pred, label="Longitud predicha", marker="o")
    # plt.title(f"Longitud real vs predicha para {callsign_to_plot} (30 segundos)")
    # plt.xlabel("Timestamp")
    # plt.ylabel("Longitud (°)")
    # plt.xticks(rotation=45)
    # plt.ylim([8.325, 8.36])
    # plt.xlim([timestamps.iloc[225], timestamps.iloc[225] + pd.Timedelta(seconds=30)])
    # plt.legend()
    # plt.tight_layout()
    # # plt.show()

    # # 7c) Gráfica ALTITUD 
    # df_geo = pd.DataFrame({
    #     "timestamp": timestamps,
    #     "geo_real": geo_real,
    #     "geo_pred": geo_pred
    # })

    # # Índice temporal para resample
    # df_geo.set_index("timestamp", inplace=True)

    # # Submuestreo cada 5 segundos con media
    # df_geo_resampled = df_geo.resample("5S").mean().dropna().reset_index()

    # # Columnas finales para graficar
    # ts_geo     = df_geo_resampled["timestamp"]
    # geo_real_s = df_geo_resampled["geo_real"]
    # geo_pred_s = df_geo_resampled["geo_pred"]

    # # Gráfica
    # plt.figure(figsize=(12, 6))
    # plt.plot(ts_geo, geo_real_s, label="Altitud real")
    # plt.plot(ts_geo, geo_pred_s, label="Altitud predicha", marker="o")
    # plt.title(f"Altitud real vs predicha para {callsign_to_plot} (muestreo cada 5 segundos)")
    # plt.xlabel("Timestamp")
    # plt.ylabel("Geoaltitude (m)")
    # plt.xticks(rotation=45)
    # plt.xlim([timestamps.iloc[0], timestamps.iloc[0] + pd.Timedelta(minutes=12)])
    # plt.legend()
    # plt.tight_layout()
    # # plt.show()

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

#Guardar en archivo CSV
output_path = f"resultados_lstm_{callsign_to_plot}.csv"
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
    # plt.show()


# resultados = []  # Lista donde guardaremos los resultados por callsign

# # Todos los callsigns únicos
# callsigns = experimento.df["callsign"].dropna().unique()

# for callsign in callsigns:
#     data_cs = experimento.df[experimento.df["callsign"] == callsign].reset_index(drop=True)

#     if len(data_cs) < experimento.lookback + experimento.lookforward:
#         continue  # Ignora si hay pocos datos

#     try:
#         X_cs, y_cs = create_windows_by_callsign(
#             data_cs,
#             experimento.lookback,
#             experimento.lookforward,
#             experimento.numeric_features,
#             experimento.categoric_features,
#             experimento.objective_features,
#             callsign_column="callsign"
#         )

#         ds_cs = tf.data.Dataset.from_tensor_slices(X_cs).batch(batch_size)
#         preds_cs = experimento.model.predict(ds_cs, verbose=0)

#         # Desescalado
#         preds2d = preds_cs.reshape(-1, len(experimento.objective_features))
#         y2d = y_cs.reshape(-1, len(experimento.objective_features))

#         preds_unsc = experimento.scaler_objective.inverse_transform(preds2d)
#         y_unsc = experimento.scaler_objective.inverse_transform(y2d)

#         # Métricas por variable
#         mae = np.mean(np.abs(preds_unsc - y_unsc), axis=0)
#         rmse = np.sqrt(np.mean((preds_unsc - y_unsc)**2, axis=0))

#         lat_mean = np.mean(y_true_unscaled[:, experimento.objective_features.index("latitude")])
#         lon_mean = np.mean(y_true_unscaled[:, experimento.objective_features.index("longitude")])
#         alt_mean = np.mean(y_true_unscaled[:, experimento.objective_features.index("geoaltitude")])

#         resultados.append({
#             "callsign": callsign,
#             "mae_latitude": mae[experimento.objective_features.index("latitude")],
#             "mae_longitude": mae[experimento.objective_features.index("longitude")],
#             "mae_geoaltitude": mae[experimento.objective_features.index("geoaltitude")],
#             "rmse_latitude": rmse[experimento.objective_features.index("latitude")],
#             "rmse_longitude": rmse[experimento.objective_features.index("longitude")],
#             "rmse_geoaltitude": rmse[experimento.objective_features.index("geoaltitude")],

#             # NUEVOS: valores reales medios
#             "mean_latitude": lat_mean,
#             "mean_longitude": lon_mean,
#             "mean_geoaltitude": alt_mean
#         })

#     except Exception as e:
#         print(f"Error con callsign {callsign}: {e}")

# import pandas as pd

# df_res = pd.DataFrame(resultados)


# df_res["error_lat_pct"] = 100 * df_res["mae_latitude"] / df_res["mean_latitude"]
# df_res["error_lon_pct"] = 100 * df_res["mae_longitude"] / df_res["mean_longitude"]
# df_res["error_alt_pct"] = 100 * df_res["mae_geoaltitude"] / df_res["mean_geoaltitude"]

# # Evitar divisiones por cero o infinitos
# df_res.replace([np.inf, -np.inf], np.nan, inplace=True)

# # Media sin unidades
# df_res["error_pct_total"] = df_res[["error_lat_pct", "error_lon_pct", "error_alt_pct"]].mean(axis=1)

# # Mostrar los mejores y peores vuelos según % de error total
# print("\n🟢 Mejores 5:")
# print(df_res.nsmallest(5, "error_pct_total"))

# print("\n🔴 Peores 5:")
# print(df_res.nlargest(5, "error_pct_total"))

# # Guardar el DataFrame
# df_res.to_csv("errores_por_callsign.csv", index=False)

