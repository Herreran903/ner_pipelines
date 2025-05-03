"""
plot_utils.py

Este módulo proporciona funciones para visualizar métricas clave durante
el entrenamiento y evaluación de modelos de reconocimiento de entidades (NER).

Incluye:
- plot_training_history(): gráfica de pérdida y validación por época.
- plot_multiple_histories(): comparación de `loss` entre varios experimentos.
- plot_f1_per_epoch(): evolución del F1-score por época.
- plot_multiple_f1(): comparación de F1-score entre experimentos.

Estas visualizaciones son útiles para evaluar el aprendizaje, detectar sobreajuste
y analizar el rendimiento de distintas configuraciones de entrenamiento.
"""

import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt  # type: ignore

# Configurar logger si aún no está definido
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def plot_multiple_histories(
    histories: List[Dict[str, List[float]]],
    labels: Optional[List[str]] = None,
    title: str = "Comparación de pérdidas entre experimentos",
) -> None:
    """
    Grafica múltiples curvas de pérdida de entrenamiento (`loss`) para comparar distintos experimentos.

    Esta función es útil para analizar visualmente el comportamiento del entrenamiento en diferentes
    configuraciones (por ejemplo, combinaciones de `batch_size`, `epochs`, embeddings, etc.).
    Permite comparar la velocidad de convergencia, estabilidad y posibles indicios de sobreajuste.

    Args:
        histories (List[Dict[str, List[float]]]): Lista de historiales devueltos por `model.fit()` (atributo `.history`).
            Cada historial debe contener la clave `'loss'` con la lista de pérdidas por época.
        labels (List[str], optional): Nombres descriptivos para cada curva. Si no se proporciona o si la longitud
            no coincide con la de `histories`, se usarán etiquetas genéricas ("Experimento 1", "Experimento 2", ...).
        title (str, optional): Título que se mostrará en la figura. Default: "Comparación de pérdidas entre experimentos".

    Returns:
        None. Muestra el gráfico comparativo directamente mediante `matplotlib`.

    Example:
        >>> plot_multiple_histories(
                histories=[history1, history2],
                labels=["Con embeddings", "Sin embeddings"],
                title="Impacto de embeddings en la convergencia"
            )

    Notes:
        - Si algún historial no contiene la clave `'loss'`, será omitido con una advertencia.
        - Las curvas se trazan con marcadores para facilitar la comparación por época.
        - Este gráfico solo incluye la pérdida de entrenamiento; si deseas comparar `val_loss`,
          puedes extender la función para incluirla.
    """
    if not histories:
        logger.warning(
            "No se proporcionaron historiales de entrenamiento para graficar."
        )
        return

    if labels and len(labels) != len(histories):
        logger.warning(
            "La longitud de `labels` no coincide con `histories`. Se usarán etiquetas genéricas."
        )
        labels = [f"Experimento {i+1}" for i in range(len(histories))]

    if labels is None:
        labels = [f"Experimento {i+1}" for i in range(len(histories))]

    plt.figure(figsize=(10, 6))

    for i, history in enumerate(histories):
        loss = history.get("loss")
        if loss is None:
            logger.warning("Historial %d no contiene 'loss', se omitirá.", i + 1)
            continue
        plt.plot(loss, label=labels[i], marker="o")

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_training_history(
    history: Dict[str, List[float]], title: str = "Training History"
) -> None:
    """
    Grafica múltiples curvas de pérdida de entrenamiento (`loss`) para comparar experimentos.

    Esta función permite visualizar cómo evoluciona la pérdida durante el entrenamiento en
    diferentes ejecuciones, facilitando el análisis comparativo entre configuraciones como
    batch size, tamaño de embedding, número de unidades LSTM, etc.

    Args:
        histories (List[Dict[str, List[float]]]): Lista de historiales retornados por `model.fit()` (history.history).
            Cada diccionario debe contener al menos la clave `'loss'`.
        labels (List[str], optional): Nombres personalizados para cada experimento. Si no se proporcionan,
            se asignan etiquetas genéricas como "Experimento 1", "Experimento 2", etc.
        title (str, optional): Título del gráfico. Default: "Comparación de pérdidas entre experimentos".

    Returns:
        None. Muestra un gráfico comparativo en pantalla.

    Example:
        >>> plot_multiple_histories([h1, h2], labels=["con Word2Vec", "sin embeddings"])

    Notes:
        - Se omiten curvas cuyo historial no contenga la clave 'loss'.
        - La comparación solo se hace sobre la pérdida de entrenamiento (`loss`), no sobre `val_loss`.
    """
    if not history:
        logger.warning("No se proporcionó historial de entrenamiento.")
        return

    loss = history.get("loss")
    val_loss = history.get("val_loss")

    if loss is None:
        logger.warning("El historial no contiene 'loss'.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(loss, label="Train Loss", marker="o")
    if val_loss is not None:
        plt.plot(val_loss, label="Validation Loss", marker="o")

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_f1_per_epoch(
    f1_scores: List[float], title: str = "F1-score por época"
) -> None:
    """
    Grafica la evolución del F1-score a lo largo de las épocas de entrenamiento.

    Esta función permite observar cómo varía el rendimiento del modelo en términos de F1-score
    durante el entrenamiento, lo cual es útil para detectar sobreajuste, subentrenamiento o
    falta de convergencia.

    Args:
        f1_scores (List[float]): Lista de F1-score calculado por época.
            Debe ser una lista de floats entre 0 y 1.
        title (str, optional): Título que se mostrará en la figura. Default: "F1-score por época".

    Returns:
        None. La función muestra el gráfico directamente usando matplotlib.

    Example:
        >>> f1_scores = [0.71, 0.78, 0.82, 0.84, 0.85]
        >>> plot_f1_per_epoch(f1_scores, title="F1-score con Word2Vec")

    Notes:
        - Para utilizar esta función es necesario calcular manualmente el F1 por época,
          almacenándolo después de cada validación.
        - El eje Y está restringido al rango [0, 1] para mantener la comparabilidad.
    """
    if not f1_scores:
        logger.warning("No se proporcionaron F1-scores.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(f1_scores, label="F1-score", color="green", marker="o")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_multiple_f1(
    f1_lists: List[List[float]],
    labels: Optional[List[str]] = None,
    title: str = "Comparación de F1-score entre experimentos",
) -> None:
    """
    Grafica y compara la evolución del F1-score por época en múltiples experimentos de entrenamiento.

    Esta función es útil para analizar el rendimiento relativo entre distintos modelos
    o configuraciones de entrenamiento (como diferentes valores de batch size, arquitectura,
    embeddings, etc.). Permite evaluar estabilidad, convergencia y sobreajuste a lo largo del tiempo.

    Args:
        f1_lists (List[List[float]]): Lista de listas de F1-score por época.
            Cada sublista representa una corrida de entrenamiento distinta.
        labels (List[str], optional): Nombres descriptivos para cada experimento.
            Si no se proporciona o no coincide en longitud, se asignan etiquetas genéricas.
        title (str, optional): Título del gráfico. Default: "Comparación de F1-score entre experimentos".

    Returns:
        None. Muestra una figura con múltiples curvas de F1-score.

    Example:
        >>> plot_multiple_f1(
                [[0.81, 0.84, 0.85], [0.79, 0.83, 0.87]],
                labels=["con Word2Vec", "sin embeddings"]
            )

    Notes:
        - Para almacenar estas curvas por época, es necesario calcular y guardar el F1 en cada `epoch`.
        - La función asume que los valores de F1 están en el rango [0, 1].
    """
    if not f1_lists:
        logger.warning("No se proporcionaron F1-scores.")
        return

    if labels and len(labels) != len(f1_lists):
        logger.warning(
            "`labels` no coincide con `f1_lists`. Se usarán etiquetas genéricas."
        )
        labels = [f"Experimento {i+1}" for i in range(len(f1_lists))]

    if labels is None:
        labels = [f"Experimento {i+1}" for i in range(len(f1_lists))]

    plt.figure(figsize=(10, 6))
    for i, f1 in enumerate(f1_lists):
        plt.plot(f1, label=labels[i], marker="o")

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
