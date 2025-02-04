import pandas as pd
import torch

from util import artifacts, constants, cache
import models
import models.query_embedder, models.query_projector, models.vectors

MAX_RESULTS = 5

def search(query: str):
    query_projector = models.query_projector.Model()

    query_state_dict = artifacts.load_artifact('query-projector-weights', 'model')

    query_projector.load_state_dict(query_state_dict)
    query_projector.eval()

    rows = pd.read_csv(constants.RESULTS_PATH)

    query_embeddings = models.query_embedder.get_embeddings_for_query(query)
    query_embeddings = torch.tensor(query_embeddings)

    encoded_query = query_projector(query_embeddings)

    def get_similarity(row):
        # use unsqueeze because cosine_similarity expects two 2d tensors
        encoded_doc = cache.vectors.get(f"encoded_docs:{row['doc_ref']}")
        doc_2d = torch.tensor(encoded_doc).unsqueeze(0)
        query_2d = encoded_query.unsqueeze(0)
        similarity = torch.nn.functional.cosine_similarity(doc_2d, query_2d).item()
        return (row['doc_ref'], similarity)

    similarities = rows.apply(get_similarity, axis=1).tolist()
    similarities.sort(reverse=True, key=lambda x: x[1])

    doc_results = [doc for (doc, similarity) in similarities[:MAX_RESULTS]]

    return doc_results
