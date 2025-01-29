from torch import nn

WINDOW_SIZE = 2
EMBEDDING_DIM = 50
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.003
VOCAB_SIZE = 50000
TRAINING_DATA_SIZE = 1000000
HACKER_NEWS_RATIO = 0.2

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, x):
        x = self.embeddings(x)  # (batch_size, embedding_dim)
        x = self.linear(x)      # (batch_size, vocab_size)
        return x
    