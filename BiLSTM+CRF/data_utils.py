"""
data_utils.py

Este módulo contiene funciones de preprocesamiento para datos anotados en formato BIO/CoNLL,
comúnmente utilizados en tareas de reconocimiento de entidades nombradas (NER).

Incluye:
- load_bio_corpus(): carga robusta de archivos BIO o CoNLL, tolerante a errores.
- build_vocab(): construcción de vocabulario de tokens.
- build_label_map(): construcción de vocabulario de etiquetas BIO.
- encode_sentences(): codificación de oraciones como índices numéricos.
- encode_labels(): codificación de etiquetas como índices.
- pad_sequences_custom(): padding manual de secuencias sin dependencias externas.
- get_idx2word(): inversión de vocabulario para obtener palabra a partir de índice.
- get_idx2tag(): inversión de vocabulario de etiquetas para obtener etiqueta a partir de índice.

Estas utilidades permiten preparar de forma estructurada datasets clínicos o generales
para su uso en modelos de secuencia como BiLSTM+CRF.
"""

from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np  # type: ignore

from config import PAD_TOKEN, UNK_TOKEN


def load_bio_corpus(
    path: Union[str, Path], verbose: bool = True
) -> Tuple[List[List[str]], List[List[str]], List[Tuple[int, str]]]:
    """
    Carga un corpus en formato BIO/CoNLL y lo estructura como listas de tokens y etiquetas por oración.

    Este lector es tolerante a errores comunes:
    - Ignora comentarios (líneas que empiezan por `#`) y marcadores de documentos (como `-DOCSTART-`).
    - Solo requiere que cada línea tenga al menos dos columnas separadas por espacio.
    - Usa la primera columna como token y la última como etiqueta (BIO, BILOU, etc.).
    - Agrupa tokens en oraciones separadas por líneas vacías.

    Es ideal para datasets como `conll2002`, `conll2003`, o cualquier dataset clínico anotado en el mismo formato.

    Args:
        path (Union[str, Path]): Ruta del archivo `.bio` o `.conll` a cargar.
        verbose (bool, optional): Si es True, imprime advertencias sobre líneas mal formateadas.
            También notifica cuántas líneas fueron ignoradas. Default es True.

    Returns:
        Tuple[List[List[str]], List[List[str]], List[Tuple[int, str]]]:
            - `sentences` (List[List[str]]): Lista de oraciones, cada una como una lista de tokens (palabras).
            - `labels` (List[List[str]]): Lista de secuencias de etiquetas correspondientes a las oraciones.
            - `bad_lines` (List[Tuple[int, str]]): Lista de tuplas con número de línea y contenido de aquellas líneas que no se pudieron interpretar correctamente (por falta de columnas o errores de formato).

    Example:
        Si el archivo contiene:

            Paciente B-PER
            masculino I-PER

            Tiene O
            hipertensión B-ENFERMEDAD

        El resultado será:

            sentences = [['Paciente', 'masculino'], ['Tiene', 'hipertensión']]
            labels    = [['B-PER', 'I-PER'], ['O', 'B-ENFERMEDAD']]
            bad_lines = []

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta dada.
    """
    sents, labels, bad_lines = [], [], []
    tokens, tags = [], []

    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()

            # Separador de oraciones
            if not line:
                if tokens:
                    sents.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
                continue

            # Ignorar líneas de metadatos o comentarios
            if line.startswith(("#", "-DOCSTART-")):
                continue

            # Particionar columnas (token, ... , tag)
            parts = line.split()
            if len(parts) < 2:
                bad_lines.append((lineno, raw))
                continue

            token, tag = parts[0], parts[-1]
            tokens.append(token)
            tags.append(tag)

    # Captura última oración si el archivo no terminó con línea en blanco
    if tokens:
        sents.append(tokens)
        labels.append(tags)

    # Mostrar advertencia si hay líneas mal formateadas
    if verbose and bad_lines:
        print(
            f"[data_utils] Aviso: {len(bad_lines)} líneas ignoradas por formato incorrecto."
        )

    return sents, labels, bad_lines


def build_vocab(
    sentences: List[List[str]], pad_token: str = PAD_TOKEN, unk_token: str = UNK_TOKEN
) -> Dict[str, int]:
    """
    Construye un diccionario de vocabulario a partir de un corpus tokenizado.

    Esta función recibe una lista de oraciones tokenizadas (listas de palabras)
    y genera un mapeo único de palabra → índice entero, asignando primero
    los tokens especiales para padding y desconocidos (`<PAD>`, `<UNK>`).

    El vocabulario resultante se utiliza comúnmente para codificar oraciones
    como secuencias numéricas que puedan ser procesadas por redes neuronales.

    Args:
        sentences (List[List[str]]): Lista de oraciones, cada una como una lista de tokens.
        pad_token (str, optional): Token reservado para padding. Por defecto "<PAD>".
        unk_token (str, optional): Token reservado para palabras fuera de vocabulario. Por defecto "<UNK>".

    Returns:
        Dict[str, int]: Diccionario que asigna a cada palabra un índice único. Los índices 0 y 1 están reservados para `pad_token` y `unk_token`, respectivamente.

    Example:
        >>> sentences = [["paciente", "presenta", "tumor"], ["historia", "clínica"]]
        >>> vocab = build_vocab(sentences)
        >>> vocab["paciente"]
        2
        >>> vocab["<PAD>"]
        0
        >>> vocab["<UNK>"]
        1

    Notes:
        - Este vocabulario no realiza ningún tipo de normalización (como minúsculas o eliminación de puntuación).
        - Palabras duplicadas se ignoran; solo se asigna un índice por palabra única.
        - El orden de asignación depende del orden en que aparecen las palabras al iterar el corpus.

    See Also:
        - `encode_sentences`: para convertir tokens a índices usando este vocabulario.
        - `build_label_map`: para construir un vocabulario de etiquetas BIO.
    """
    word2idx = {pad_token: 0, unk_token: 1}
    for word in Counter(chain.from_iterable(sentences)):
        word2idx.setdefault(word, len(word2idx))
    return word2idx


def build_label_map(labels: List[List[str]]) -> Dict[str, int]:
    """
    Construye un diccionario que asigna un índice único a cada etiqueta BIO.

    Esta función toma una lista de listas de etiquetas (como 'O', 'B-DIAG', 'I-TRATAMIENTO', etc.)
    y construye un mapeo ordenado alfabéticamente de etiqueta → índice. Es utilizado para convertir
    secuencias de etiquetas categóricas en representaciones numéricas que puedan ser usadas por modelos de ML.

    Args:
        labels (List[List[str]]): Lista de listas de etiquetas BIO correspondientes a cada token en cada oración.

    Returns:
        Dict[str, int]: Diccionario que asigna a cada etiqueta un índice entero único.

    Example:
        >>> etiquetas = [["O", "B-DIAG", "I-DIAG"], ["O", "B-TRATAMIENTO"]]
        >>> build_label_map(etiquetas)
        {'B-DIAG': 0, 'B-TRATAMIENTO': 1, 'I-DIAG': 2, 'O': 3}

    Notes:
        - El orden de los índices depende del orden alfabético de las etiquetas.
        - No se añade explícitamente una etiqueta para padding (como "<PAD>"); debe añadirse manualmente si se necesita.
        - Esta función no comprueba duplicados porque `set()` elimina automáticamente repeticiones.

    See Also:
        - `encode_labels`: para transformar etiquetas usando este mapeo.
        - `build_vocab`: para hacer lo mismo con tokens.
    """
    label_set = sorted(set(chain.from_iterable(labels)))
    return {label: idx for idx, label in enumerate(label_set)}


def encode_sentences(
    sentences: List[List[str]], word2idx: Dict[str, int]
) -> List[List[int]]:
    """
    Convierte una lista de oraciones tokenizadas en secuencias numéricas usando un vocabulario.

    Para cada palabra en cada oración, se asigna su índice correspondiente según el vocabulario.
    Si una palabra no se encuentra en el vocabulario (`word2idx`), se le asigna el índice
    correspondiente al token desconocido (`<UNK>`), que debe estar definido previamente
    dentro del diccionario `word2idx`.

    Args:
        sentences (List[List[str]]): Lista de oraciones, cada una como una lista de tokens (palabras).
        word2idx (Dict[str, int]): Diccionario que mapea palabras a índices enteros.
            Debe contener al menos el token especial "<UNK>".

    Returns:
        List[List[int]]: Lista de oraciones codificadas como listas de enteros.

    Example:
        >>> vocab = {'<PAD>': 0, '<UNK>': 1, 'paciente': 2, 'tumor': 3}
        >>> sentences = [['paciente', 'presenta', 'tumor']]
        >>> encode_sentences(sentences, vocab)
        [[2, 1, 3]]  # "presenta" no está en el vocabulario → <UNK> (índice 1)

    Notes:
        - La función no realiza padding. Para eso debe usarse `pad_sequences_custom`.
        - El token `<UNK>` debe existir en el vocabulario, o se lanzará una excepción.
        - Este paso es fundamental antes de alimentar un modelo que trabaja con embeddings o capas de entrada numérica.

    See Also:
        - `build_vocab`: para crear el diccionario de tokens.
        - `encode_labels`: para transformar etiquetas BIO en índices numéricos.
        - `pad_sequences_custom`: para aplicar padding a las secuencias codificadas.
    """
    return [
        [word2idx.get(word, word2idx["<UNK>"]) for word in sent] for sent in sentences
    ]


def encode_labels(
    labels: List[List[str]], label_map: Dict[str, int]
) -> List[List[int]]:
    """
    Convierte una lista de etiquetas categóricas (BIO, BILOU, etc.) en listas de índices numéricos.

    Utiliza un diccionario `label_map` que asigna a cada etiqueta su índice correspondiente.
    Esta función es útil para transformar etiquetas como 'O', 'B-DIAG', 'I-TRATAMIENTO' en
    vectores numéricos que puedan ser utilizados como salida supervisada durante el entrenamiento
    de modelos de secuencias.

    Args:
        labels (List[List[str]]): Lista de secuencias de etiquetas (por ejemplo, en formato BIO),
            donde cada sublista corresponde a las etiquetas de una oración.
        label_map (Dict[str, int]): Diccionario que mapea cada etiqueta a un índice único.

    Returns:
        List[List[int]]: Lista de secuencias de etiquetas codificadas como índices enteros.

    Example:
        >>> etiquetas = [["O", "B-TRATAMIENTO"], ["B-DIAG", "I-DIAG", "O"]]
        >>> label_map = {"B-DIAG": 0, "B-TRATAMIENTO": 1, "I-DIAG": 2, "O": 3}
        >>> encode_labels(etiquetas, label_map)
        [[3, 1], [0, 2, 3]]

    Notes:
        - La función no valida si todas las etiquetas existen en `label_map`. Si hay alguna ausente, lanzará `KeyError`.
        - El orden de los índices dependerá del `label_map`, que usualmente se genera con `build_label_map`.

    See Also:
        - `build_label_map`: para construir el diccionario de etiquetas.
        - `encode_sentences`: para codificar tokens como índices usando `word2idx`.
    """
    return [[label_map[label] for label in label_seq] for label_seq in labels]


def pad_sequences_custom(
    sequences: List[List[int]], maxlen: int, pad_value: int = 0
) -> np.ndarray:
    """
    Aplica padding manual a las secuencias numéricas de longitud variable.

    Esta implementación no depende de TensorFlow ni Keras y devuelve un arreglo NumPy.
    Es útil para prototipos ligeros, procesamiento previo o cuando se desea portabilidad completa.

    Args:
        sequences (List[List[int]]): Lista de secuencias (listas de enteros) con longitud variable.
        maxlen (int): Longitud fija deseada para todas las secuencias tras aplicar padding o truncamiento.
        pad_value (int, optional): Valor que se usará para rellenar las posiciones vacías. Default es 0.

    Returns:
        np.ndarray: Matriz 2D de shape `(n_samples, maxlen)` con padding aplicado a la derecha.

    Example:
        >>> pad_sequences_custom([[1, 2, 3], [4, 5]], maxlen=4)
        array([[1, 2, 3, 0],
               [4, 5, 0, 0]])

    Notes:
        - Trunca por la derecha si una secuencia excede `maxlen`.
        - Aplica padding al final (por la derecha).
        - Usa `np.full` para inicializar la matriz con el valor de padding.
        - Ideal para usar en preprocesamiento o como alternativa a `tensorflow.keras.utils.pad_sequences` si no se quiere depender de TensorFlow.

    See Also:
        - `pad_sequences_tf` en módulos con TensorFlow para integración nativa en pipelines Keras.
    """
    padded = np.full((len(sequences), maxlen), pad_value, dtype=int)
    for i, seq in enumerate(sequences):
        trunc = seq[:maxlen]
        padded[i, : len(trunc)] = trunc
    return padded


def get_idx2word(word2idx: Dict[str, int]) -> Dict[int, str]:
    """
    Invierte un diccionario de vocabulario para obtener índice → palabra.

    Args:
        word2idx (dict): Diccionario palabra → índice.

    Returns:
        dict: Diccionario índice → palabra.
    """
    return {i: w for w, i in word2idx.items()}


def get_idx2tag(tag2idx: Dict[str, int]) -> Dict[int, str]:
    """
    Invierte un diccionario de etiquetas para obtener índice → etiqueta.

    Args:
        tag2idx (dict): Diccionario etiqueta → índice.

    Returns:
        dict: Diccionario índice → etiqueta.
    """
    return {i: t for t, i in tag2idx.items()}
