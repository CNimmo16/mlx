import gensim.downloader
import numpy as np
import random

from util import mini
from models import query_embedder

word_vectors = None

def get_random_vec():
    return np.float32(np.random.random(size=query_embedder.EMBEDDING_DIM))

def get_vecs():
    global word_vectors

    if mini.is_quick_vecs():
        word_vectors = {
            'the': get_random_vec(),
            'a': get_random_vec()
        }
        word_vectors["<UNK>"] = get_random_vec()
        return word_vectors

    if not word_vectors:
        print("Downloading word vectors...")
        word_vectors = gensim.downloader.load('word2vec-google-news-300')
        word_vectors["<UNK>"] = get_random_vec()
        print("Done")
    return word_vectors
