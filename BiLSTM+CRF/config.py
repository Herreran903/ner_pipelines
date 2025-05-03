"""
config.py

Este módulo contiene constantes globales reutilizables para todo el pipeline BiLSTM+CRF
aplicado a tareas de reconocimiento de entidades clínicas (NER).

Incluye:
- Tokens especiales para padding y palabras desconocidas.
- Índices estándar (PAD_IDX, UNK_IDX) que deben coincidir con el vocabulario.
- Hiperparámetros por defecto del modelo y entrenamiento.
- Rutas a los archivos del dataset y embeddings preentrenados.
- Combinaciones de hiperparámetros para experimentación automática.
"""

from pathlib import Path

# === Tokens especiales ===
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# === Índices especiales (estos deben alinearse con el vocabulario creado) ===
PAD_IDX = 0
UNK_IDX = 1

# === Hiperparámetros generales ===
MAX_LEN = 100
DEFAULT_EMB_DIM = 300
DEFAULT_LSTM_UNITS = 128
DEFAULT_DROPOUT = 0.25
DEFAULT_TRAINABLE_EMB = True
DEFAULT_USE_MASKING = True
BATCH_SIZE = 32
EPOCHS = 30

# === Configuración de experimentos ===
EARLY_STOP_PATIENCE = 3
VALIDATION_SPLIT = 0.1

# === Rutas (ajústalas a tu entorno local o Colab) ===
BASE_DIR = Path("data")
TRAIN_PATH = BASE_DIR / "training.bio"
VALID_PATH = BASE_DIR / "validation.bio"
TEST_PATH = BASE_DIR / "testing.bio"

# === Embeddings preentrenados ===
W2V_MODEL_PATH = Path("embeddings") / "es_w2v_s300.bin"  # Ejemplo

# === Combinaciones de hiperparámetros a evaluar ===
EXPERIMENT_GRID = [
    {"batch_size": 16, "epochs": 10},
    {"batch_size": 16, "epochs": 20},
    {"batch_size": 32, "epochs": 10},
    {"batch_size": 32, "epochs": 20},
    {"batch_size": 64, "epochs": 10},
    {"batch_size": 64, "epochs": 20},
]
