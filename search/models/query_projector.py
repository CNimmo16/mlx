import torch
from models import query_embedder, doc_embedder

QUERY_HIDDEN_LAYER_DIMENSIONS = [128, 64]

OUTPUT_DIMENSION = 256

DROPOUT = 0.1

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.tower = torch.nn.Sequential(
            torch.nn.Linear(query_embedder.EMBEDDING_DIM, QUERY_HIDDEN_LAYER_DIMENSIONS[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(QUERY_HIDDEN_LAYER_DIMENSIONS[0], QUERY_HIDDEN_LAYER_DIMENSIONS[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(QUERY_HIDDEN_LAYER_DIMENSIONS[1], OUTPUT_DIMENSION)
        )
        self.dropout = torch.nn.Dropout(DROPOUT)

    def forward(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        return self.tower(query_embeddings)
