import torch
import pandas as pd
import numpy as np

import models
import models.query_embedder, models.doc_embedder
from util import devices

CHUNK_SIZE = 10

device = devices.get_device()

class TwoTowerDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.prepped = {}

    def __len__(self):
        return len(self.data.index)
    
    def __get_irrelevant_doc_embedding(self, row):
        for i in range(0, 100):
            other_row = self.data.sample(1).iloc[0]
            if (other_row['query'] != row['query']) and (other_row['doc_ref'] != row['doc_ref']):
                return models.doc_embedder.get_embeddings_for_doc(other_row['doc_text'])
        raise ValueError("No non-relevant result found in random 100 row sample. Weird.")

    def __prepare_row(self, row):
        return pd.Series({
            'query_embeddings': models.query_embedder.get_embeddings_for_query(row['query']),
            'relevant_doc_embeddings': models.doc_embedder.get_embeddings_for_doc(row['doc_text']),
            'irrelevant_doc_embeddings': self.__get_irrelevant_doc_embedding(row)
        })
    
    def __get_chunk(self, chunk_idx: int):
        if chunk_idx not in self.prepped:
            rows = self.data[chunk_idx * CHUNK_SIZE:(chunk_idx + 1) * CHUNK_SIZE]
            self.prepped[chunk_idx] = rows.swifter.progress_bar(False).apply(self.__prepare_row, axis=1)
        return self.prepped[chunk_idx]

    def __getitem__(self, idx: int) -> dict[str, str]:
        chunk_idx = idx // CHUNK_SIZE

        chunk = self.__get_chunk(chunk_idx)

        idx_in_chunk = idx % CHUNK_SIZE
        
        return chunk.iloc[idx_in_chunk]

def get_padded(batch: list, field: str):
    embeddings = [torch.tensor(item[field]).to(device) for item in batch]

    padded_query_embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True, padding_value=0).float()
    query_embedding_lengths = torch.tensor([len(x) for x in embeddings])

    return padded_query_embeddings, query_embedding_lengths    

def collate_two_tower_batch(batch: list):
    query_embeddings, query_embedding_lengths = get_padded(batch, 'query_embeddings')
    relevant_doc_embeddings, relevant_doc_embedding_lengths = get_padded(batch, 'relevant_doc_embeddings')
    irrelevant_doc_embeddings, irrelevant_doc_embedding_lengths = get_padded(batch, 'irrelevant_doc_embeddings')

    return {
        'query_embeddings': query_embeddings,
        'query_embedding_lengths': query_embedding_lengths,
        'relevant_doc_embeddings': relevant_doc_embeddings,
        'relevant_doc_embedding_lengths': relevant_doc_embedding_lengths,
        'irrelevant_doc_embeddings': irrelevant_doc_embeddings,
        'irrelevant_doc_embedding_lengths': irrelevant_doc_embedding_lengths
    }
