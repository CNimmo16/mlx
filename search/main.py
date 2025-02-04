import pandas as pd
import torch

from util import artifacts, constants, cache
from models import query_embedder, two_towers, vectors

MAX_RESULTS = 5

def search(query: str):
    model = two_towers.Model()

    state_dict = artifacts.load_artifact('two-towers-weights')

    model.load_state_dict(state_dict)
    model.eval()

    rows = pd.read_csv(constants.RESULTS_PATH)

    query_embeddings = query_embedder.get_embeddings_for_query(query)
    query_embeddings = torch.tensor(query_embeddings)

    encoded_query = model.encode_query(query_embeddings)

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

def cli():
    vectors.get_vecs()

    query = input("Enter a query: ")

    results = search(query)

    print(results)

    cli()

cli()
