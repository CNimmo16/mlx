import torch
import os

MINIMODE = int(os.environ.get('FULLRUN', '0')) == 0

print(f"MINI MODE: {MINIMODE}")

WINDOW_SIZE = 2
EMBEDDING_DIM = 50
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.003
MAX_VOCAB_SIZE = 50000
TRAINING_DATA_SIZE = 1000 if MINIMODE else 5000000
HACKER_NEWS_RATIO = 0.2

class Model(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Model, self).__init__()
        self.embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.linear = torch.nn.Linear(in_features=embedding_dim, out_features=vocab_size, bias=False)
    
    def forward(self, words):
        # Get the embedding for each word
        embedded = self.embeddings(words)

        # perform the linear transformation back to full vocab size
        logits = self.linear(embedded)
        
        return logits
