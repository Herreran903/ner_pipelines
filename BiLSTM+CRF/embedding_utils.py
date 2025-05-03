"""
embedding_utils.py

Este módulo permite cargar modelos de embeddings preentrenados (Word2Vec)
y construir matrices de embeddings alineadas con un vocabulario (`word2idx`).

Incluye:
- Carga de modelos en formatos `.bin`, `.kv`, `.txt`.
- Generación de matriz `embedding_matrix` con vectores en orden indexado.
"""

import logging
from typing import Dict

import numpy as np  # type: ignore
from gensim.models import KeyedVectors  # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_word2vec_model(path: str) -> KeyedVectors:
    """
    Carga un modelo Word2Vec desde un archivo.

    Detecta automáticamente si es formato binario o texto. Acepta formatos:
    - `.bin` (binario)
    - `.txt` (texto plano)
    - `.kv` (KeyedVectors de Gensim)

    Args:
        path (str): Ruta al archivo del modelo.

    Returns:
        gensim.models.KeyedVectors: Modelo cargado listo para consulta.
    """
    if path.endswith(".kv"):
        model = KeyedVectors.load(path)
    elif path.endswith(".bin"):
        model = KeyedVectors.load_word2vec_format(path, binary=True)
    else:
        model = KeyedVectors.load_word2vec_format(path, binary=False)

    logger.info("Modelo Word2Vec cargado con %d vectores.", len(model))
    return model


def build_embedding_matrix(
    word2idx: Dict[str, int],
    word2vec_model: KeyedVectors,
    emb_dim: int,
    pad_token: str = "<PAD>",
    unk_token: str = "<UNK>",
) -> np.ndarray:
    """
    Construye una matriz de embeddings a partir de un modelo Word2Vec alineado con `word2idx`.

    Cada fila de la matriz corresponde al vector del índice en `word2idx`.
    Las palabras que no están en el modelo recibirán un vector aleatorio
    muestreado de una distribución normal (media 0, desviación estándar 0.01).

    Args:
        word2idx (Dict[str, int]): Vocabulario que mapea tokens a índices.
        word2vec_model (KeyedVectors): Modelo cargado con `load_word2vec_model`.
        emb_dim (int): Dimensión de los vectores en el modelo.
        pad_token (str, optional): Token para padding. Recibe vector de ceros. Default: "<PAD>".
        unk_token (str, optional): Token para palabras desconocidas. Recibe vector aleatorio. Default: "<UNK>".

    Returns:
        np.ndarray: Matriz de shape (len(word2idx), emb_dim) lista para `Embedding(..., weights=[...])`.

    Raises:
        ValueError: Si `emb_dim` no coincide con los vectores del modelo cargado.
    """
    if word2vec_model.vector_size != emb_dim:
        raise ValueError(
            f"El modelo tiene dimensión {word2vec_model.vector_size}, se esperaba {emb_dim}"
        )

    embedding_matrix = np.random.normal(0, 0.01, size=(len(word2idx), emb_dim)).astype(
        np.float32
    )

    for word, idx in word2idx.items():
        if word == pad_token:
            embedding_matrix[idx] = np.zeros((emb_dim,), dtype=np.float32)
        elif word in word2vec_model:
            embedding_matrix[idx] = word2vec_model[word]

    logger.info("Matriz de embeddings creada con shape %s.", embedding_matrix.shape)
    return embedding_matrix
