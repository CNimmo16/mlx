import nltk
import numpy as np
from models import vectors

EMBEDDING_DIM = 300

def get_embeddings_for_doc(doc: str) -> list:
    word_vectors = vectors.get_vecs()
    tokens = nltk.word_tokenize(doc)
    embeddings = [word_vectors[token] for token in tokens if token in word_vectors]
    ret = np.mean(embeddings, axis=0)
    return ret
