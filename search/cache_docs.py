import pandas as pd
import torch
import swifter

from util import artifacts, constants, cache
import models
import models.doc_embedder, models.doc_projector

def cache_doc_encodings():
    torch.no_grad()

    data = pd.read_csv(constants.RESULTS_PATH)

    doc_projector = models.doc_projector.Model()

    doc_state_dict = artifacts.load_artifact('doc-projector-weights', 'model')

    doc_projector.load_state_dict(doc_state_dict)
    doc_projector.eval()

    def get_doc_encoding(row):
        doc_embeddings = models.doc_embedder.get_embeddings_for_doc(row['doc_text'])
        doc_embeddings = torch.tensor(doc_embeddings)
        encoded = doc_projector(doc_embeddings).detach().tolist()
        cache.vectors.set(f"encoded_docs:{row['doc_ref']}", encoded)
        return encoded

    data.apply(get_doc_encoding, axis=1)

cache_doc_encodings()
