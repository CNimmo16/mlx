import torch
from models import query_embedder

QUERY_HIDDEN_LAYER_DIMENSION = 128

OUTPUT_DIMENSION = 256

DROPOUT = 0.1

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.rnn = torch.nn.LSTM(
            input_size=query_embedder.EMBEDDING_DIM,
            hidden_size=QUERY_HIDDEN_LAYER_DIMENSION,
            batch_first=True
        )
    
        # Final projection layer
        self.project = torch.nn.Linear(QUERY_HIDDEN_LAYER_DIMENSION, OUTPUT_DIMENSION)

    def forward(self, query_embeddings: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        max_len = torch.max(lengths).item()
        expected_shape = [max_len, query_embedder.EMBEDDING_DIM]
        try:
            assert(list(query_embeddings.shape)[1:] == expected_shape)
        except:
            raise ValueError(f"Embedding shape [?, {list(query_embeddings.shape)}] did not match expected shape [?, {expected_shape}]")

        # Pack padded input
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(query_embeddings, lengths, batch_first=True, enforce_sorted=False)
        
        # Pass through LSTM
        packed_output, (h_n, c_n) = self.rnn(packed_x)

        final_layer_output = h_n.squeeze(0)
        
        # Unpack the output back to padded
        # output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        projected = self.project(final_layer_output)

        # mean = torch.mean(projected, dim=1)

        return projected, (h_n, c_n)
