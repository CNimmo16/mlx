import torch

WINDOW_SIZE = 2
EMBEDDING_DIM = 50
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.003
VOCAB_SIZE = 50000
TRAINING_DATA_SIZE = 1000000
HACKER_NEWS_RATIO = 0.2

class Model(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Model, self).__init__()
        self.embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.linear = torch.nn.Linear(in_features=embedding_dim, out_features=vocab_size)

        # Creates a Sigmoid activation function, which squashes outputs to the range (0,1). Used to calculate probabilities.
        self.sig = torch.nn.Sigmoid()
    
    # def forward(self, input, targets, random):
    def forward(self, input, targets):
        # x = self.embeddings(x)  # (batch_size, embedding_dim)
        # x = self.linear(x)      # (batch_size, vocab_size)
        # return x

        # Look up embeddings for the input words (center words).
        embedding_for_input = self.embeddings(input)

        # Get embeddings for the target words (positive samples) from the Linear layer's weight matrix.
        targets_for_input = self.linear.weight[targets]

        # # Get embeddings for the random words (negative samples) from the Linear layer's weight matrix.
        # rnd = self.linear.weight[random]

        # Compute the dot product between target word embeddings and input embeddings.
        # `bmm` performs batch matrix multiplication. `unsqueeze(-1)` adds a dimension for the dot product.
        output_dot = torch.bmm(targets_for_input, embedding_for_input.unsqueeze(-1)).squeeze()

        # Compute the dot product between random word embeddings and input embeddings.
        # rnd = torch.bmm(rnd, emb.unsqueeze(-1)).squeeze()

        # Apply the sigmoid function to convert scores into probabilities.
        output_prob = self.sig(output_dot)
        # rnd = self.sig(rnd)

        # Compute the positive loss: Negative log likelihood of the positive samples.
        positive_loss = -output_prob.log().mean()

        # # Compute the negative loss: Negative log likelihood of the negative samples.
        # # Add a small constant (10^-3) to avoid log(0).
        # ngt = -(1 - rnd + 10**(-3)).log().mean()

        # Return the combined loss (positive loss + negative loss).
        # return pst + ngt
        return positive_loss
    