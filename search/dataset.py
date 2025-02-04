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

def collate_two_tower_batch(batch: list):
    query_embeddings = np.array([item['query_embeddings'] for item in batch])
    relevant_doc_embeddings = np.array([item['relevant_doc_embeddings'] for item in batch])
    irrelevant_doc_embeddings = np.array([item['irrelevant_doc_embeddings'] for item in batch])

    return {
        'query_embeddings': torch.tensor(query_embeddings).to(device),
        'relevant_doc_embeddings': torch.tensor(relevant_doc_embeddings).to(device),
        'irrelevant_doc_embeddings': torch.tensor(irrelevant_doc_embeddings).to(device)
    }
    
    # Tokenize all texts
    query_tokens = [tokenizer.tokenize(q) for q in queries]
    pos_tokens = [tokenizer.tokenize(p) for p in positives]
    neg_tokens = [tokenizer.tokenize(n) for n in negatives]
    
    # Get max lengths
    max_query_len = max(len(t) for t in query_tokens)
    max_pos_len = max(len(t) for t in pos_tokens)
    max_neg_len = max(len(t) for t in neg_tokens)
    
    # Pad sequences
    pad_id = tokenizer.special_tokens['<PAD>']
    
    padded_queries = [
        t + [pad_id] * (max_query_len - len(t)) for t in query_tokens
    ]
    padded_positives = [
        t + [pad_id] * (max_pos_len - len(t)) for t in pos_tokens
    ]
    padded_negatives = [
        t + [pad_id] * (max_neg_len - len(t)) for t in neg_tokens
    ]
    
    # Convert to tensors
    return {
        'query_ids': torch.tensor(padded_queries),
        'query_lengths': torch.tensor([len(t) for t in query_tokens]),
        'positive_ids': torch.tensor(padded_positives),
        'positive_lengths': torch.tensor([len(t) for t in pos_tokens]),
        'negative_ids': torch.tensor(padded_negatives),
        'negative_lengths': torch.tensor([len(t) for t in neg_tokens])
    }