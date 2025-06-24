import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.api.layers import LSTM, Dense, Dropout, Bidirectional
from keras.api.models import Sequential
from keras.api.callbacks import CSVLogger, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, LabelEncoder
tf.config.run_functions_eagerly(True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_windows_by_callsign(df, lookback, lookforward, numeric_features, categoric_features, objective_features, *, callsign_column="callsign"):
    """
    Se espera que se pasen los siguientes argumentos:
      - df: DataFrame de entrada.
      - lookback: número de pasos de entrada.
      - lookforward: número de pasos a predecir.
      - numeric_features: lista de nombres de columnas numéricas.
      - categoric_features: lista de nombres de columnas categóricas.
      - objective_features: lista de nombres de columnas objetivo.
    El parámetro callsign_column se pasa como keyword-only.
    Para la entrada X se usa la unión de numeric_features y categoric_features.
    """
    X_windows = []
    y_windows = []
    # Combinar features de entrada: numéricas y categóricas.
    features_input = numeric_features + categoric_features
    for cs, group in df.groupby(callsign_column):
        group = group.sort_values(by="timestamp")
        total_steps = lookback + lookforward
        if len(group) < total_steps:
            continue
        for i in range(len(group) - total_steps + 1):
            window = group.iloc[i:i+total_steps]
            X = window[features_input].values[:lookback]
            y = window[objective_features].values[-lookforward:]
            # Asegurarse de que X y y tengan al menos 2 dimensiones
            if X.ndim == 1:
                X = np.expand_dims(X, axis=0)
            if y.ndim == 1:
                y = np.expand_dims(y, axis=0)
            X_windows.append(X)
            y_windows.append(y)
    
    X_windows = np.array(X_windows)
    y_windows = np.array(y_windows)
    return X_windows, y_windows


# Clase base con los métodos esenciales para entrenamiento y predicción.
class Experiment:
    def __init__(self, lookback, lookforward, shift, batch_size, features):
        self.lookback = lookback
        self.lookforward = lookforward
        self.shift = shift
        self.batch_size = batch_size
        self.features = features
        self.numeric_feat = features.get('numeric', [])
        self.categoric_feat = features.get('categoric', [])
        self.objective_feat = features.get('objective', [])
        self.num_features = len(self.numeric_feat) + len(self.categoric_feat)
        self.model = None
        self.trained_epochs = 0
        # Se define la ruta de guardado del modelo en una carpeta local "models"
        self.model_path = Path("models")
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.model_path_save = self.model_path / 'ep{epoch:03d}_loss{loss:.4f}_val{val_loss:.4f}.h5'
        self.model_path_best = self.model_path / 'best.h5'
        self.model_path_last = self.model_path / 'last.h5'
        self.model_path_log  = self.model_path / 'log.csv'

    def init_model(self):
        raise NotImplementedError

    def _format_data(self, dataset):
        """
        En esta versión, se asume que los datos ya están en (X, y).
        """
        return dataset

    def load_model(self, name='last'):
        model_file = self.model_path / f'{name}.h5'
        if not model_file.exists():
            print(f"No se encontró el modelo {model_file}. Entrenando un modelo nuevo.")
            return
        self.model = tf.keras.models.load_model(model_file)

        encoder_path = self.model_path / f'encoder_{self.num_features}.joblib'
        scaler_path  = self.model_path / f'scaler_{self.num_features}.joblib'
        if encoder_path.exists() and scaler_path.exists():
            self.encoders = joblib.load(encoder_path)
            self.scaler   = joblib.load(scaler_path)
        else:
            print("No se encontraron archivos de encoder/scaler. Entrenando un modelo nuevo.")

    def _init_callbacks(self):
        modelCheckpoint = ModelCheckpoint(
            self.model_path_save,
            monitor='val_loss',
            verbose=0,
            mode='auto',
            save_best_only=False
        )
        modelCheckpointBest = ModelCheckpoint(
            self.model_path_best,
            monitor='val_loss',
            verbose=0,
            mode='auto',
            save_best_only=True
        )
        modelCheckpointLast = ModelCheckpoint(
            self.model_path_last,
            monitor='val_loss',
            verbose=0,
            mode='auto',
            save_best_only=False
        )
        csvLogger = CSVLogger(self.model_path_log, append=True)
        self.callbacks = [modelCheckpoint, modelCheckpointBest, modelCheckpointLast, csvLogger]

    def train(self, epochs, add_callbacks=None):
        # Los métodos _load_data son implementados en la subclase para CSV.
        train_dataset = self._load_data('train', randomize=True)
        val_dataset = self._load_data('val', randomize=False)
        try:
            logs = pd.read_csv(self.model_path_log)
            if self.trained_epochs != logs.shape[0]:
                logs[logs.epoch < self.trained_epochs].to_csv(self.model_path_log, index=False)
        except FileNotFoundError:
            pass

        # Usa self en lugar de "experimento"
        test_dataset = self._load_data("test", randomize=True).batch(self.batch_size)
        X_batch, y_batch = next(iter(test_dataset))

        preds = self.model(X_batch)

        h = self.model.fit(
            x=train_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE),
            validation_data=val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE),
            epochs=epochs,
            verbose=1,
            callbacks=self.callbacks + (add_callbacks if add_callbacks else []),
            initial_epoch=self.trained_epochs
        )
        self.trained_epochs = pd.read_csv(self.model_path_log).epoch.max() + 1
        return h

    def predict_trajectory(self, data: pd.DataFrame) -> pd.DataFrame:
        dataset = self._format_data(data)
        predictions = self.model.predict(dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE), verbose=0)
        predictions = predictions.reshape(-1, len(self.objective_feat))
        unsc_predictions = self.scaler.inverse_transform(
            np.concatenate([np.zeros((predictions.shape[0], self.num_features)), predictions], axis=1)
        )[:, -len(self.objective_feat):]
        return pd.DataFrame(unsc_predictions, columns=self.objective_feat)

# Clase que implementa un modelo LSTM vanilla.
class ExperimentVanilla(Experiment):
    def __init__(self, lookback, sampling, model_config, features, lookforward=1, shift=-1, model_type=None):
        self.model_type = model_type if model_type else 'LSTM'
        self.n_units = model_config.get('n_units')
        self.act_function = model_config.get('act_function')
        self.loss_function = model_config.get('loss_function', 'mean_absolute_error')
        self.optimizer = model_config.get('optimizer', 'adam')
        self.sampling = sampling
        self.airport = ""
        self.months = ""
        self.batch_size = model_config.get('batch_size', 128)
        # Se define la ruta de guardado en la carpeta "models"
        self.model_path = Path("models") / f'{self.model_type}_s{sampling}_lb{lookback}_u{self.n_units}'
        self.model_path.mkdir(parents=True, exist_ok=True)
        super().__init__(lookback, lookforward, shift, self.batch_size, features)
        self.init_model()
        self._write_config()

    def init_model(self, add_metrics=None):
        self.model = Sequential([
        Bidirectional(LSTM(self.n_units, activation=self.act_function), input_shape=(self.lookback, self.num_features)),
        # Dropout(0.2),
        Dense(self.lookforward * len(self.objective_feat)),
            tf.keras.layers.Reshape((self.lookforward, len(self.objective_feat)))
        ])
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['mean_squared_error'] + (add_metrics if add_metrics else []))
        self._init_callbacks()

    def _write_config(self):
        experiment_config = {
            "model_type": self.model_type,
            "num_units": self.n_units,
            "activation_function": self.act_function,
            "loss_function": self.loss_function,
            "batch_size": self.batch_size,
            "lookback": self.lookback,
            "lookforward": self.lookforward,
            "shift": self.shift,
            "sampling": self.sampling,
            "airport": self.airport,
            "features": self.features
        }
        with open(self.model_path / 'experiment_config.json', 'w+') as output_file:
            json.dump(experiment_config, output_file)

    def _format_data(self, dataset):
        # Se asume que el dataset ya está en formato (X, y)
        return dataset

# Clase adaptada para trabajar con CSV.
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class ExperimentCSVVanilla(ExperimentVanilla):
    def __init__(self, csv_path, lookback, lookforward, model_config, features, batch_size):
        self.csv_path = csv_path
        self.numeric_features = features.get("numeric", [])
        self.categoric_features = features.get("categoric", [])
        self.objective_features = features.get("objective", [])

        # 1) Leer el CSV sin cabecera, especificando los nombres de columna
        #    Si el separador es la coma (,) y las comillas son dobles ("), pandas debería detectarlo.
        df = pd.read_csv(
            csv_path,
            header=None,  # No hay fila con nombres de columna
            names=["TIME", "ICAO", "LAT", "LONG", "HEADING", "VELOCITY", "VERTICAL_RATE", "GEOALTITUD"]
        )

        # 2) Renombrar las columnas a los nombres que usabas en el código original
        df.rename(columns={
            "TIME": "timestamp",
            "ICAO": "callsign",
            "LAT": "latitude",
            "LONG": "longitude",
            "HEADING": "track",
            "VELOCITY": "groundspeed",
            "VERTICAL_RATE": "vertical_rate",
            "GEOALTITUD": "geoaltitude"
        }, inplace=True)

        # 3) Convertir 'timestamp' a datetime (asumiendo que es epoch en segundos)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")

        # 3.1) Filtrar los datos para tomar solo registros cada 10 segundos
        # df = df[df["timestamp"].dt.second % 5 == 0]

        # 4) Eliminar filas con timestamp o callsign vacío
        df.dropna(subset=["callsign", "timestamp"], inplace=True)

        # 5) Ordenar por callsign y timestamp
        df.sort_values(by=["callsign", "timestamp"], inplace=True)

        # 6) Conservar solo las columnas relevantes (se incluyen también las categóricas, si existen)
        columns_to_keep = ["timestamp", "callsign"] + self.numeric_features + self.categoric_features
        df = df[columns_to_keep]

        # 7) Eliminar filas que tengan NaN en las columnas objetivo
        #    (en tu ejemplo: latitude, longitude, geoaltitude)
        df.dropna(subset=self.objective_features, inplace=True)

        # 8) Convertir las columnas numéricas a float
        df[self.numeric_features] = df[self.numeric_features].apply(pd.to_numeric, errors='coerce')
        #    Si quedaran NaN tras la conversión, los descartamos también
        df.dropna(subset=self.numeric_features, inplace=True)

        # (Opcional) Imprimir estadísticas para verificar los datos
        print(df[self.numeric_features].describe())
        print("NaNs por columna:\n", df.isna().sum())
        print("Infinitos en las columnas numéricas:\n", np.isinf(df[self.numeric_features]).sum())

        # 9) Antes de escalar numeric, guardo copia de los originales
        orig_obj = df[self.objective_features].copy()

        # 10) Escalar variables numéricas (incluye lat/lon)
        self.scaler_numeric = StandardScaler()
        df[self.numeric_features] = self.scaler_numeric.fit_transform(df[self.numeric_features])

        # 11) Ahora ajuste de objetivo SOBRE LOS VALORES ORIGINALES
        self.scaler_objective = StandardScaler()
        self.scaler_objective.fit(orig_obj)
        df[self.objective_features] = self.scaler_objective.transform(orig_obj)

        # 11) Guardar el CSV procesado para revisión (opcional)
        processed_csv_path = "processed_data.csv"
        df.to_csv(processed_csv_path, index=False)
        print(f"CSV procesado guardado en '{processed_csv_path}'.")

        # 12) Guardar el DataFrame final en self.df
        self.df = df.reset_index(drop=True)

        # 13) Llamar al constructor de la clase padre
        super().__init__(lookback, sampling=1, model_config=model_config, features=features,
                         lookforward=lookforward, shift=0, model_type="CSV_LSTM")
        self.batch_size = batch_size
        self.init_model()

    def _load_data(self, dataset, randomize):
        total_len = len(self.df)
        train_end = int(0.7 * total_len)
        val_end = int(0.85 * total_len)

        if dataset == 'train':
            df_subset = self.df.iloc[:train_end]
        elif dataset == 'val':
            df_subset = self.df.iloc[train_end:val_end]
        else:  # dataset == 'test'
            df_subset = self.df.iloc[val_end:]

        X, y = create_windows_by_callsign(
            df_subset,
            self.lookback,
            self.lookforward,
            self.numeric_features,
            self.categoric_features,
            self.objective_features,
            callsign_column="callsign"
        )

        if randomize:
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]

        print(f"Dataset={dataset}, X.shape={X.shape}, y.shape={y.shape}")
        return tf.data.Dataset.from_tensor_slices((X, y))

    
    def _format_data(self, data):
        X, y = create_windows_by_callsign(
            data,
            self.lookback,
            self.lookforward,
            self.numeric_features,
            self.categoric_features,
            self.objective_features,
            callsign_column="callsign"
        )
        return tf.data.Dataset.from_tensor_slices((X, y))
    
    def predict_trajectory(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data[self.numeric_features] = self.scaler_numeric.transform(data[self.numeric_features])
        for col in self.categoric_features:
            data[col] = self.encoders[col].transform(data[col])
        X, _ = create_windows_by_callsign(
            data,
            self.lookback,
            self.lookforward,
            self.numeric_features,
            self.categoric_features,
            self.objective_features,
            callsign_column="callsign"
        )
        dataset = tf.data.Dataset.from_tensor_slices(X)
        predictions = self.model.predict(dataset.batch(self.batch_size), verbose=0)
        predictions = predictions.reshape(-1, len(self.objective_features))
        unscaled_predictions = self.scaler_objective.inverse_transform(predictions)
        return pd.DataFrame(unscaled_predictions, columns=self.objective_features)