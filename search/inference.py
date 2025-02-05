import pandas as pd
import torch

from util import artifacts, constants, chroma, devices
import models
import dataset
import models.query_embedder, models.query_projector, models.vectors

MAX_RESULTS = 5

device = devices.get_device()

query_projector = None
rows = None

def load_model_and_rows():
    global query_projector, rows
    if query_projector is None or rows is None:
        query_projector = models.query_projector.Model().to(device)

        query_state_dict = artifacts.load_artifact('query-projector-weights', 'model')

        query_projector.load_state_dict(query_state_dict)
        query_projector.eval()

        rows = pd.read_csv(constants.RESULTS_PATH)

    return query_projector, rows

def get_random_query():
    query_projector, rows = load_model_and_rows()

    query = rows.sample(1).iloc[0]['query']

    return query

def search(query: str):
    query_projector, rows = load_model_and_rows()

    query_embeddings = models.query_embedder.get_embeddings_for_query(query)

    batch = [query_embeddings]

    padded_embeddings, lengths = dataset.pad_batch_values(batch)

    encoded_query_batch, _ = query_projector(padded_embeddings, lengths)
    
    encoded_query_batch_list = encoded_query_batch.detach().tolist()

    collection = chroma.client.get_collection(name="docs")
    nearest_docs = collection.query(
        query_embeddings=encoded_query_batch_list,
        n_results=5,
    )
    nearest_doc_refs = nearest_docs['ids'][0]
    nearest_docs = [rows[rows['doc_ref'] == id].iloc[0].to_dict() for id in nearest_doc_refs]

    return nearest_docs
