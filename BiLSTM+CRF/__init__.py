"""
ner_pipeline_bilstm_crf/__init__.py

Este paquete contiene herramientas para el entrenamiento, evaluación y experimentación
con modelos BiLSTM+CRF aplicados al reconocimiento de entidades clínicas (NER).

Submódulos:
- config: Configuración global (tokens, hiperparámetros, rutas).
- data_utils: Carga y procesamiento de datos en formato BIO.
- embedding_utils: Carga de modelos Word2Vec y construcción de matrices de embeddings.
- model_utils: Definición del modelo BiLSTM+CRF.
- train_eval_utils: Entrenamiento, evaluación y experimentación.

Uso típico:
    from ner_pipeline import config, model_utils, data_utils
"""

from . import config
from .data_utils import (
    build_label_map,
    build_vocab,
    encode_labels,
    encode_sentences,
    load_bio_corpus,
    pad_sequences_custom,
)
from .embedding_utils import build_embedding_matrix, load_word2vec_model
from .model_utils import build_bilstm_crf_model, get_model_summary
from .train_eval_utils import evaluate_model, run_experiments, train_model
