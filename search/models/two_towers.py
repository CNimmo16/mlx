import torch
from models import query_embedder, doc_embedder

QUERY_HIDDEN_LAYER_DIMENSIONS = [128, 64]
DOC_HIDDEN_LAYER_DIMENSIONS = [128, 64]

OUTPUT_DIMENSION = 256

MARGIN = 0.2
LEARNING_RATE = 0.0001
DROPOUT = 0.1

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.query_tower = torch.nn.Sequential(
            torch.nn.Linear(query_embedder.EMBEDDING_DIM, OUTPUT_DIMENSION),
            # torch.nn.Linear(query_embedder.EMBEDDING_DIM, QUERY_HIDDEN_LAYER_DIMENSIONS[0]),
            # torch.nn.ReLU(),
            # torch.nn.Linear(QUERY_HIDDEN_LAYER_DIMENSIONS[0], QUERY_HIDDEN_LAYER_DIMENSIONS[1]),
            # torch.nn.ReLU(),
            # torch.nn.Linear(QUERY_HIDDEN_LAYER_DIMENSIONS[1], OUTPUT_DIMENSION)
        )
        self.doc_tower = torch.nn.Sequential(
            torch.nn.Linear(query_embedder.EMBEDDING_DIM, OUTPUT_DIMENSION),
            # torch.nn.Linear(doc_embedder.EMBEDDING_DIM, DOC_HIDDEN_LAYER_DIMENSIONS[0]),
            # torch.nn.ReLU(),
            # torch.nn.Linear(DOC_HIDDEN_LAYER_DIMENSIONS[0], DOC_HIDDEN_LAYER_DIMENSIONS[1]),
            # torch.nn.ReLU(),
            # torch.nn.Linear(DOC_HIDDEN_LAYER_DIMENSIONS[1], OUTPUT_DIMENSION)
        )
        self.dropout = torch.nn.Dropout(DROPOUT)

    def encode_query(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        return self.query_tower(query_embeddings)
    
    def encode_doc(self, doc_embeddings: torch.Tensor) -> torch.Tensor:
        return self.doc_tower(doc_embeddings)
