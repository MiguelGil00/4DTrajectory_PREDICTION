import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.api.utils import register_keras_serializable
from keras.api.layers import (
    Input, Dense, LayerNormalization, Dropout,
    MultiHeadAttention, Add, Reshape, Layer
)
from keras.api.models import Model
from keras.api.callbacks import CSVLogger, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, LabelEncoder

tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Positional Encoding Layer ---
@tf.keras.utils.register_keras_serializable()
class AddPositionalEncoding(Layer):
    def __init__(self, lookback, num_features, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.num_features = num_features

    def build(self, input_shape):
        # Learnable positional embeddings
        self.pos_emb = self.add_weight(
            name='pos_emb',
            shape=(self.lookback, self.num_features),
            initializer='random_normal',
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.pos_emb

@register_keras_serializable()
class GetItem(Layer):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, inputs):
        return inputs[:, self.index]

    def get_config(self):
        config = super().get_config()
        config.update({"index": self.index})
        return config

# --- Data windowing utility (unchanged) ---
def create_windows_by_callsign(df, lookback, lookforward, numeric_features, categoric_features, objective_features, *, callsign_column="callsign"):
    X_windows = []
    y_windows = []
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
            if X.ndim == 1:
                X = np.expand_dims(X, axis=0)
            if y.ndim == 1:
                y = np.expand_dims(y, axis=0)
            X_windows.append(X)
            y_windows.append(y)
    X_windows = np.array(X_windows)
    y_windows = np.array(y_windows)
    print("Shape of X_windows:", X_windows.shape)
    print("Shape of y_windows:", y_windows.shape)
    return X_windows, y_windows

# --- Base experiment class ---
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
        self.model_path = Path("models")
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.model_path_save = self.model_path / 'ep{epoch:03d}_loss{loss:.4f}_val{val_loss:.4f}.h5'
        self.model_path_best = self.model_path / 'best.h5'
        self.model_path_last = self.model_path / 'last.h5'
        self.model_path_log  = self.model_path / 'log.csv'

    def init_model(self):
        raise NotImplementedError

    def _init_callbacks(self):
        self.callbacks = [
            ModelCheckpoint(self.model_path_save, monitor='val_loss', save_best_only=False),
            ModelCheckpoint(self.model_path_best, monitor='val_loss', save_best_only=True),
            ModelCheckpoint(self.model_path_last, monitor='val_loss', save_best_only=False),
            CSVLogger(self.model_path_log, append=True)
        ]

    def load_model(self, name='last'):
        model_file = self.model_path / f'{name}.h5'
        if not model_file.exists():
            print(f"No se encontró modelo {model_file}, se entrenará uno nuevo.")
            return
        self.model = tf.keras.models.load_model(model_file)
        enc_path = self.model_path / f'encoder_{self.num_features}.joblib'
        sca_path = self.model_path / f'scaler_{self.num_features}.joblib'
        if enc_path.exists():
            self.scaler_numeric = joblib.load(enc_path)
        if sca_path.exists():
            self.scaler_objective = joblib.load(sca_path)

    def train(self, epochs, add_callbacks=None):
        train_ds = self._load_data('train', randomize=True)
        val_ds = self._load_data('val', randomize=False)
        try:
            logs = pd.read_csv(self.model_path_log)
            if self.trained_epochs != logs.shape[0]:
                logs[logs.epoch < self.trained_epochs].to_csv(self.model_path_log, index=False)
        except FileNotFoundError:
            pass
        self._init_callbacks()
        history = self.model.fit(
            x=train_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE),
            validation_data=val_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE),
            epochs=epochs,
            callbacks=self.callbacks + (add_callbacks or []),
            initial_epoch=self.trained_epochs,
            verbose=1
        )
        # Save scalers for inference
        joblib.dump(self.scaler_numeric, self.model_path / f'encoder_{self.num_features}.joblib')
        joblib.dump(self.scaler_objective, self.model_path / f'scaler_{self.num_features}.joblib')
        self.trained_epochs += len(history.history['loss'])
        return history

    def predict_trajectory(self, data: pd.DataFrame) -> pd.DataFrame:
        ds = self._format_data(data)
        preds = self.model.predict(ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE), verbose=0)
        preds = preds.reshape(-1, len(self.objective_feat))
        unsc = self.scaler_objective.inverse_transform(preds)
        return pd.DataFrame(unsc, columns=self.objective_feat)

# --- TFT Experiment class ---
class ExperimentTFT(Experiment):
    def __init__(self, lookback, model_config, features, lookforward=1, shift=0):
        self.model_config = model_config
        self.model_type = 'TFT'
        super().__init__(lookback, lookforward, shift,
                         batch_size=model_config.get('batch_size', 128),
                         features=features)
        self._build_model()
        self._write_config()

    def _write_config(self):
        cfg = { 'model_type': self.model_type, **self.model_config,
                'lookback': self.lookback, 'lookforward': self.lookforward,
                'features': self.features }
        with open(self.model_path / 'experiment_config.json','w') as f:
            json.dump(cfg, f)

    def _build_model(self):
        num_heads = self.model_config.get('num_heads', 4)
        ff_dim = self.model_config.get('ff_dim', 128)
        num_enc = self.model_config.get('num_encoder_layers', 2)
        num_dec = self.model_config.get('num_decoder_layers', 1)
        dropout_rate = self.model_config.get('dropout', 0)

        inputs = Input(shape=(self.lookback, self.num_features), name='time_series_input')
        x = inputs
        x = AddPositionalEncoding(self.lookback, self.num_features)(x)

        for i in range(num_enc):
            attn = MultiHeadAttention(num_heads=num_heads,
                                      key_dim=self.num_features)(x, x)
            x = LayerNormalization(epsilon=1e-5)(Add()([x, Dropout(dropout_rate)(attn)]))
            ffn = Dense(ff_dim, activation='relu')(x)
            ffn = Dense(self.num_features)(ffn)
            x = LayerNormalization(epsilon=1e-5)(Add()([x, Dropout(dropout_rate)(ffn)]))

        y = x
        seq_len = self.lookback
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        for j in range(num_dec):
            a1 = MultiHeadAttention(num_heads=num_heads, key_dim=self.num_features)(y, y, attention_mask=mask)
            y = LayerNormalization(epsilon=1e-5)(Add()([y, Dropout(dropout_rate)(a1)]))
            a2 = MultiHeadAttention(num_heads=num_heads, key_dim=self.num_features)(y, x)
            y = LayerNormalization(epsilon=1e-5)(Add()([y, Dropout(dropout_rate)(a2)]))
            ffn2 = Dense(ff_dim, activation='relu')(y)
            ffn2 = Dense(self.num_features)(ffn2)
            y = LayerNormalization(epsilon=1e-5)(Add()([y, Dropout(dropout_rate)(ffn2)]))

        last = y[:, -1, :]
        out = Dense(self.lookforward * len(self.objective_feat), name='pred_dense')(last)
        out = Reshape((self.lookforward, len(self.objective_feat)))(out)

        self.model = Model(inputs=inputs, outputs=out, name='TFT_Model')
        self.model.compile(
            loss=self.model_config.get('loss','mae'),
            optimizer=self.model_config.get('optimizer','adam'),
            metrics=self.model_config.get('metrics',['mse'])
        )

    def init_model(self):
        pass

class ExperimentCSVTFT(ExperimentTFT):
    def __init__(self, csv_path, lookback, lookforward, model_config, features, batch_size=128):
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

        # Llamar padre con batch_size en model_config
        model_config["batch_size"] = batch_size
        super().__init__(lookback, model_config, features,
                         lookforward=lookforward, shift=0)
        self.batch_size = batch_size

    def _load_data(self, dataset, randomize):
        total = len(self.df)
        tr, va = int(0.7 * total), int(0.85 * total)
        sub = {'train': self.df.iloc[:tr], 'val': self.df.iloc[tr:va], 'test': self.df.iloc[va:]}[dataset]
        X, y = create_windows_by_callsign(sub, self.lookback, self.lookforward,
                                         self.numeric_feat, self.categoric_feat,
                                         self.objective_feat)
        if randomize:
            idx = np.random.permutation(len(X))
            X, y = X[idx], y[idx]
        return tf.data.Dataset.from_tensor_slices((X, y))

    def _format_data(self, data):
        X, _ = create_windows_by_callsign(data, self.lookback, self.lookforward,
                                          self.numeric_feat, self.categoric_feat,
                                          self.objective_feat)
        return tf.data.Dataset.from_tensor_slices(X)
