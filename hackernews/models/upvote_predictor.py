import torch
import torch.nn as nn
import os

MINIMODE = int(os.environ.get('FULLRUN', '0')) == 0

print(f"MINI MODE: {MINIMODE}")

HIDDEN_LAYER_DIMENSIONS = [256, 128]
EPOCHS = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 64

class Model(nn.Module):
    def __init__(self, embedding_dim):
        """
        Args:
            embedding_dim (int): Dimension of the title embeddings
            hidden_dims (list): List of hidden layer dimensions
        """
        super().__init__()
        
        # Define the network architecture
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim + 1, HIDDEN_LAYER_DIMENSIONS[0]),  # +1 for karma
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_DIMENSIONS[0], HIDDEN_LAYER_DIMENSIONS[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_DIMENSIONS[1], 1)
        )
        
    def forward(self, title_embedding, karma):
        """
        Args:
            title_embedding (torch.Tensor): Title embedding tensor of shape (batch_size, embedding_dim)
            karma (torch.Tensor): User karma tensor of shape (batch_size, 1)
            
        Returns:
            torch.Tensor: Predicted number of upvotes (batch_size)
        """
        # Concatenate features along the feature dimension
        x = torch.cat([title_embedding, karma], dim=1)
        
        # Forward pass through network
        x = self.layers(x)
        
        # Remove extra dimension for regression output
        return x.squeeze(-1)
