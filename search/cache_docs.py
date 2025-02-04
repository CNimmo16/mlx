import pandas as pd
import torch
import swifter

from util import artifacts, constants, cache
from models import two_towers, doc_embedder

def cache_doc_encodings():
    torch.no_grad()

    data = pd.read_csv(constants.RESULTS_PATH)

    model = two_towers.Model()

    state_dict = artifacts.load_artifact('two-towers-weights')

    model.load_state_dict(state_dict)
    model.eval()

    def get_doc_encoding(row):
        doc_embeddings = doc_embedder.get_embeddings_for_doc(row['doc_text'])
        doc_embeddings = torch.tensor(doc_embeddings)
        encoded = model.encode_doc(doc_embeddings).detach().tolist()
        cache.vectors.set(f"encoded_docs:{row['doc_ref']}", encoded)
        return encoded

    data.apply(get_doc_encoding, axis=1)

cache_doc_encodings()
