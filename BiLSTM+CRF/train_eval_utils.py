"""
train_eval_utils.py

Este módulo contiene funciones para entrenar, evaluar y comparar modelos BiLSTM+CRF
aplicados a tareas de reconocimiento de entidades nombradas (NER).

Incluye:
- train_model(): entrenamiento del modelo con validación y early stopping opcional.
- evaluate_model(): evaluación del modelo usando métricas de seqeval (F1-score, precision, recall).
- run_experiments(): ejecución automatizada de múltiples combinaciones de hiperparámetros.
- save_model(): guarda el modelo entrenado (Keras + CRF) en formato SavedModel.
- load_model_with_crf(): carga modelos previamente guardados con soporte para la capa CRF y su wrapper.

Estas utilidades permiten integrar fácilmente flujos de entrenamiento reproducibles,
gestión de experimentos y serialización del pipeline de NER.
"""

import logging
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from seqeval.metrics import classification_report, f1_score  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

from model_wrapper import ModelWithCRFLoss

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from config import BATCH_SIZE, EARLY_STOP_PATIENCE, EPOCHS


def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    early_stop: bool = True,
    patience: int = EARLY_STOP_PATIENCE,
) -> Dict[str, float]:
    """
    Entrena un modelo Keras con validación opcional y early stopping.

    Args:
        model: Modelo Keras compilado.
        X_train (np.ndarray): Datos de entrada de entrenamiento (padded).
        y_train (np.ndarray): Etiquetas codificadas.
        X_val (np.ndarray): Datos de validación.
        y_val (np.ndarray): Etiquetas de validación.
        batch_size (int, optional): Tamaño del batch. Default: 32.
        epochs (int, optional): Número máximo de épocas. Default: 30.
        early_stop (bool, optional): Si True, usa EarlyStopping. Default: True.
        patience (int, optional): Número de épocas sin mejora para detener. Default: 3.

    Returns:
        dict: Diccionario con historial final de entrenamiento.
    """
    callbacks = []
    if early_stop:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            )
        )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
    )

    logger.info(
        "Entrenamiento finalizado. Última época: %d", len(history.history["loss"])
    )
    return history.history


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    idx2tag: Dict[int, str],
    pad_value: int = 0,
) -> Tuple[List[str], List[str], float]:
    """
    Evalúa el modelo en el conjunto de prueba usando métricas de `seqeval`.

    Args:
        model: Modelo entrenado con capa CRF.
        X_test (np.ndarray): Datos de entrada (padded).
        y_test (np.ndarray): Etiquetas verdaderas (padded).
        idx2tag (dict): Mapeo índice → etiqueta (BIO).
        pad_value (int): Valor de padding en etiquetas. Default: 0.

    Returns:
        tuple: (true_labels, pred_labels, f1_score)
    """
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)

    true_labels, pred_labels = [], []

    for true_seq, pred_seq in zip(y_test, y_pred):
        t_seq, p_seq = [], []
        for t, p in zip(true_seq, pred_seq):
            if t != pad_value:
                t_seq.append(idx2tag[t])
                p_seq.append(idx2tag[p])
        true_labels.append(t_seq)
        pred_labels.append(p_seq)

    logger.info("Evaluación completada. Generando reporte...")
    print(classification_report(true_labels, pred_labels))
    f1 = f1_score(true_labels, pred_labels)
    return true_labels, pred_labels, f1


def run_experiments(
    model_fn,
    param_grid: List[Dict],
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    idx2tag: Dict[int, str],
    pad_value: int = 0,
) -> List[Dict]:
    """
    Ejecuta múltiples configuraciones de entrenamiento para comparar desempeño.

    Args:
        model_fn (callable): Función que construye y devuelve un modelo nuevo.
        param_grid (List[Dict]): Lista de diccionarios con parámetros `batch_size`, `epochs`, etc.
        train_data (Tuple[np.ndarray, np.ndarray]): (X_train, y_train).
        val_data (Tuple[np.ndarray, np.ndarray]): (X_val, y_val).
        test_data (Tuple[np.ndarray, np.ndarray]): (X_test, y_test).
        idx2tag (Dict[int, str]): Mapeo de índice a etiqueta.
        pad_value (int, optional): Índice usado como padding. Default: 0.

    Returns:
        List[Dict]: Lista con resultados de cada configuración, incluyendo F1-score.
    """
    results = []

    for i, params in enumerate(param_grid):
        logger.info("=== Ejecutando experimento %d/%d ===", i + 1, len(param_grid))
        logger.info("Parámetros: %s", params)

        model = model_fn(**params.get("model_args", {}))
        train_model(
            model,
            *train_data,
            *val_data,
            batch_size=params.get("batch_size", 32),
            epochs=params.get("epochs", 30)
        )

        _, _, f1 = evaluate_model(
            model, *test_data, idx2tag=idx2tag, pad_value=pad_value
        )

        result = {**params, "f1_score": round(f1, 4)}
        results.append(result)

    return results


def save_model(model, path: str) -> None:
    """
    Guarda el modelo Keras (envuelto en ModelWithCRFLoss) en el formato `SavedModel`.

    Args:
        model: Instancia de ModelWithCRFLoss.
        path (str): Ruta destino donde se guardará el modelo.
    """
    model.model.save(path)
    logger.info("Modelo guardado en '%s'.", path)


def load_model_with_crf(path: str) -> ModelWithCRFLoss:
    """
    Carga un modelo BiLSTM+CRF previamente guardado en formato `SavedModel`,
    incluyendo la capa CRF de TensorFlow Addons y el wrapper ModelWithCRFLoss.

    Args:
        path (str): Ruta del directorio donde fue guardado el modelo.

    Returns:
        ModelWithCRFLoss: Modelo Keras envuelto con la clase de pérdida CRF.
    """
    from tensorflow.keras.models import load_model  # type: ignore
    from tensorflow_addons.layers import CRF  # type: ignore

    try:
        base_model = load_model(path, custom_objects={"CRF": CRF})
        wrapped_model = ModelWithCRFLoss(base_model, sparse_target=True)
        wrapped_model.compile(optimizer="adam")
        logger.info("Modelo cargado exitosamente desde '%s'.", path)
        return wrapped_model
    except Exception as e:
        logger.error("Error al cargar el modelo desde '%s': %s", path, str(e))
        raise
