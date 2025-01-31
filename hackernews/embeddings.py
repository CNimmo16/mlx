import tokenization
from util import artifacts
import pandas as pd
import torch
import numpy as np

vocab = artifacts.load_artifact('vocab')
embedding_weights = artifacts.load_artifact('embeddings')
    
def get_embeddings_for_token(token):    
    id = tokenization.getIdFromToken(vocab, token)

    return embedding_weights[id]
    
def get_embeddings_for_title(title):
    tokens = tokenization.tokenize(vocab, title)
    ids = [tokenization.getIdFromToken(vocab, token) for token in tokens if token != 'unk']

    if (len(ids) == 0):
        return pd.NA

    embeddings_for_ids = [embedding_weights[id] for id in ids]
    
    stacked_embeddings = torch.stack(embeddings_for_ids)

    # Compute the mean along the token dimension (dim=0)
    mean_embedding = np.array(torch.mean(stacked_embeddings, dim=0).tolist())

    return mean_embedding
