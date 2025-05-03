"""
model_utils.py

Este módulo contiene funciones relacionadas con la construcción del modelo BiLSTM+CRF
para tareas de reconocimiento de entidades nombradas (NER). Utiliza Keras sobre TensorFlow
y una capa CRF para modelar dependencias entre etiquetas.

Incluye:
- build_bilstm_crf_model(): construcción del modelo con embeddings opcionales.
- get_model_summary(): impresión del resumen del modelo.
- pad_sequences_tf(): padding de secuencias numéricas compatible con pipelines de Keras.
- Constantes configurables para hiperparámetros por defecto.
"""

import logging
from typing import List, Optional

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from mwrapper import ModelWithCRFLoss  # type: ignore
from tensorflow.keras.layers import Bidirectional  # type: ignore
from tensorflow.keras.layers import Input  # type: ignore
from tensorflow.keras.layers import Masking  # type: ignore
from tensorflow.keras.layers import LSTM, Embedding, SpatialDropout1D  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow_addons.layers import CRF  # type: ignore

# Configurar logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from config import (
    DEFAULT_DROPOUT,
    DEFAULT_EMB_DIM,
    DEFAULT_LSTM_UNITS,
    DEFAULT_TRAINABLE_EMB,
    DEFAULT_USE_MASKING,
)


def build_bilstm_crf_model(
    max_len: int,
    n_words: int,
    n_tags: int,
    embedding_matrix: Optional[tf.Tensor] = None,
    emb_dim: int = DEFAULT_EMB_DIM,
    lstm_units: int = DEFAULT_LSTM_UNITS,
    dropout: float = DEFAULT_DROPOUT,
    trainable_emb: bool = DEFAULT_TRAINABLE_EMB,
    use_masking: bool = DEFAULT_USE_MASKING,
) -> tf.keras.Model:
    """
    Construye un modelo BiLSTM + CRF usando TensorFlow y Keras.

    Puede utilizar una capa de embedding aleatoria o cargada previamente
    (por ejemplo, desde Word2Vec). El modelo se compila usando un wrapper
    personalizado (`ModelWithCRFLoss`) para que la capa CRF maneje el loss.

    Args:
        max_len (int): Longitud máxima de las secuencias (padded).
        n_words (int): Tamaño del vocabulario (tokens).
        n_tags (int): Número de etiquetas BIO distintas.
        embedding_matrix (Optional[tf.Tensor], optional): Matriz de embeddings preentrenados.
            Si None, se inicializa aleatoriamente. Default: None.
        emb_dim (int, optional): Dimensión de los embeddings. Ignorado si se pasa una matriz. Default: 300.
        lstm_units (int, optional): Número de unidades en la capa LSTM. Default: 128.
        dropout (float, optional): Proporción de dropout en SpatialDropout1D. Default: 0.25.
        trainable_emb (bool, optional): Si True, permite ajustar los embeddings durante el entrenamiento. Default: True.
        use_masking (bool, optional): Si True, aplica capa Masking con valor cero. Default: True.

    Returns:
        tf.keras.Model: Modelo compilado listo para entrenamiento y evaluación.

    Raises:
        AssertionError: Si `embedding_matrix` está definida pero no coincide con `n_words`.

    Example:
        >>> model = build_bilstm_crf_model(max_len=100, n_words=5000, n_tags=15)
        >>> model.summary()
    """
    if embedding_matrix is not None:
        assert (
            embedding_matrix.shape[0] == n_words
        ), f"La matriz de embeddings debe tener {n_words} filas, pero tiene {embedding_matrix.shape[0]}."

    inputs = Input(shape=(max_len,), name="tokens")

    if embedding_matrix is None:
        x = Embedding(
            input_dim=n_words, output_dim=emb_dim, mask_zero=True, name="embedding"
        )(inputs)
    else:
        x = Embedding(
            input_dim=n_words,
            output_dim=embedding_matrix.shape[1],
            weights=[embedding_matrix],
            trainable=trainable_emb,
            mask_zero=True,
            name="embedding_pretrained",
        )(inputs)

    x = SpatialDropout1D(dropout, name="spatial_dropout")(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True), name="bilstm")(x)

    if use_masking:
        x = Masking(mask_value=0.0, name="masking")(x)

    crf = CRF(n_tags, name="crf")
    outputs = crf(x)

    model = Model(inputs, outputs, name="BiLSTM_CRF")
    model = ModelWithCRFLoss(model, sparse_target=True)
    model.compile(optimizer="adam")

    logger.info(
        "Modelo BiLSTM+CRF construido con %d etiquetas y %d palabras.", n_tags, n_words
    )

    return model


def get_model_summary(model: tf.keras.Model) -> None:
    """
    Imprime el resumen de arquitectura de un modelo Keras.

    Args:
        model (tf.keras.Model): Modelo a inspeccionar.
    """
    model.summary(line_length=120)


def pad_sequences_tf(
    sequences: List[List[int]],
    maxlen: int,
    pad_value: int = 0,
    padding: str = "post",
    truncating: str = "post",
) -> np.ndarray:
    """
    Aplica padding a secuencias numéricas usando `tensorflow.keras.preprocessing.sequence.pad_sequences`.

    Esta versión es útil cuando ya se trabaja con pipelines Keras y se desea una solución estándar
    para rellenar o truncar secuencias de manera flexible.

    Args:
        sequences (List[List[int]]): Lista de secuencias numéricas (enteros).
        maxlen (int): Longitud máxima deseada. Las secuencias más largas se truncarán.
        pad_value (int, optional): Valor utilizado para rellenar. Default: 0.
        padding (str, optional): Puede ser 'pre' o 'post'. Indica si el padding se añade al inicio o al final. Default: 'post'.
        truncating (str, optional): Puede ser 'pre' o 'post'. Indica si el truncamiento se hace desde el inicio o el final. Default: 'post'.

    Returns:
        np.ndarray: Arreglo NumPy 2D con padding aplicado a todas las secuencias.

    Example:
        >>> pad_sequences_tf([[5, 6, 7], [1, 2]], maxlen=4)
        array([[5, 6, 7, 0],
               [1, 2, 0, 0]])

    See Also:
        - `pad_sequences_custom`: alternativa sin TensorFlow en `data_utils.py`.
    """
    return pad_sequences(
        sequences,
        maxlen=maxlen,
        value=pad_value,
        padding=padding,
        truncating=truncating,
    )
