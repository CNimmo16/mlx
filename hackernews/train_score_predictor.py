import torch
import pandas as pd
import os
from util import artifacts, cache
import tokenization
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sklearn
import numpy as np
from models import skipgram, upvote_predictor
import wandb
import embeddings
import swifter # do not remove - used indirectly by DataFrame.swifter
import os
import tqdm
dirname = os.path.dirname(__file__)

wandb.init(project='word2vec', name='upvote_predictor-mini' if upvote_predictor.MINIMODE else 'upvote_predictor')
wandb.config = {
    "learning_rate": upvote_predictor.LEARNING_RATE,
    "hidden_layer_dimensions": upvote_predictor.HIDDEN_LAYER_DIMENSIONS,
    "batch_size": skipgram.BATCH_SIZE,
    "epochs": skipgram.EPOCHS
}

items_table = "hacker_news.items"

vocab = artifacts.load_artifact('vocab')

print("Fetching posts from db")

hn_posts = cache.query("hn_posts_for_predictor", f"""SELECT
    title,
    karma,
    score
    FROM {items_table}
    INNER JOIN hacker_news.users u ON {items_table}.by = u.id
    WHERE type = 'story' AND title IS NOT null
    LIMIT {100 if upvote_predictor.MINIMODE else 1000000}
""")

na_posts = hn_posts[hn_posts['title'].isna()]

print(f"Dropping {len(na_posts)} posts with missing titles")

hn_posts.dropna(inplace=True)

print("Getting embeddings for titles")

hn_posts['embeddings'] = hn_posts['title'].swifter.apply(embeddings.get_embeddings_for_title)

na_embeddings = hn_posts[hn_posts['embeddings'].isna()]

print(f"Dropping {len(na_embeddings)} posts with no embeddings (no recognised words)")

hn_posts.dropna(inplace=True)

# 1. Define Dataset Class with Normalization
class PostDataset(Dataset):
    def __init__(self, title_embeddings, karma, upvotes, embed_scaler=None, karma_scaler=None):
        # Normalize features
        title_embeddings = np.vstack(np.array(title_embeddings))
        if embed_scaler is None:
            self.embed_scaler = sklearn.preprocessing.StandardScaler()
            self.title_embeds = self.embed_scaler.fit_transform(title_embeddings)

            if not upvote_predictor.MINIMODE:
                artifacts.save_artifact(self.embed_scaler, 'embed-scaler', 'model', os.path.join(dirname, 'data/embed-scaler.generated.pt'))

        else:
            self.embed_scaler = embed_scaler
            self.title_embeds = embed_scaler.transform(title_embeddings)

        karma = karma.reshape(-1, 1)
        if karma_scaler is None:
            self.karma_scaler = sklearn.preprocessing.StandardScaler()
            self.karma = self.karma_scaler.fit_transform(karma)

            if not upvote_predictor.MINIMODE:   
                artifacts.save_artifact(self.karma_scaler, 'karma-scaler', 'model', os.path.join(dirname, 'data/karma-scaler.generated.pt'))
        else:
            self.karma_scaler = karma_scaler
            self.karma = karma_scaler.transform(karma)

        log_upvotes = np.log1p(upvotes) 
        self.upvotes = log_upvotes.reshape(-1, 1)

    def __len__(self):
        return len(self.upvotes)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.title_embeds[idx], dtype=torch.float32),
            torch.tensor(self.karma[idx], dtype=torch.float32),
            torch.tensor(self.upvotes[idx], dtype=torch.float32)
        )

# 2. Data Preparation
def prepare_loaders(title_embeddings, karma, upvotes, test_size=0.2, batch_size=32):
    # Split data
    X_train, X_val, k_train, k_val, y_train, y_val = sklearn.model_selection.train_test_split(
        title_embeddings, karma, upvotes, test_size=test_size, random_state=42
    )
    
    # Create datasets with proper scaling
    train_dataset = PostDataset(X_train, k_train, y_train)
    val_dataset = PostDataset(X_val, k_val, y_val,
                             train_dataset.embed_scaler,
                             train_dataset.karma_scaler)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# 3. Training Loop with Validation
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_state_dict = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for embeddings, karma, upvotes in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            embeddings = embeddings.to(device)
            karma = karma.to(device)
            upvotes = upvotes.to(device).squeeze()
            
            optimizer.zero_grad()
            outputs = model(embeddings, karma)
            loss = criterion(outputs, upvotes)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * embeddings.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for embeddings, karma, upvotes in val_loader:
                embeddings = embeddings.to(device)
                karma = karma.to(device)
                upvotes = upvotes.to(device).squeeze()
                
                outputs = model(embeddings, karma)
                loss = criterion(outputs, upvotes)
                val_loss += loss.item() * embeddings.size(0)
        
        # Calculate epoch losses
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)

        wandb.log({'epoch': epoch + 1, 'train-loss': train_loss, 'val-loss': val_loss})
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
    
    print('Training complete')
    if not upvote_predictor.MINIMODE:
        artifacts.save_artifact(best_state_dict, 'predictor-weights', 'model', os.path.join(dirname, 'data/predictor-weights.generated.pt'))
    return model

print("Starting training")

title_embeddings = hn_posts['embeddings'].values
karma = hn_posts['karma'].values
upvotes = hn_posts['score'].values

# Prepare data loaders
train_loader, val_loader = prepare_loaders(
    title_embeddings, karma, upvotes, batch_size=upvote_predictor.BATCH_SIZE
)

# Initialize model
model = upvote_predictor.Model(embedding_dim=skipgram.EMBEDDING_DIM)

# Train model
trained_model = train_model(model, train_loader, val_loader, epochs=upvote_predictor.EPOCHS, lr=upvote_predictor.LEARNING_RATE)

wandb.finish()
