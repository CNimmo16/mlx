import torch
from models import skipgram
import pandas as pd
import os
import wandb
import joblib

import os
dirname = os.path.dirname(__file__)

api = wandb.Api()

def download_from_wandb(ref: str, file: str):
    artifact = api.artifact(f"cnimmo16/word2vec/{ref}:latest")
    dir = artifact.download(os.path.join(dirname, '../artifacts'))
    return os.path.join(dir, file)
    
def load_artifact(ref: str):
    if (ref == 'vocab'):
        vocab_path = download_from_wandb('vocab', 'vocab.generated.csv')
        return pd.read_csv(vocab_path).set_index('token')
    if (ref == 'embeddings'):
        skipgram_weights_path = download_from_wandb('skipgram-weights', 'skipgram-weights.generated.pt')
        return torch.load(skipgram_weights_path, map_location=torch.device('cpu'))['embeddings.weight']
    if (ref == 'predictor-state'):
        predictor_weights_path = download_from_wandb('predictor-weights', 'predictor-weights.generated.pt')
        return torch.load(predictor_weights_path, map_location=torch.device('cpu'))
    if (ref == 'predictor-embed-scaler'):
        return joblib.load(download_from_wandb('embed-scaler', 'embed-scaler.generated.pt'))
    if (ref == 'predictor-karma-scaler'):
        return joblib.load(download_from_wandb('karma-scaler', 'karma-scaler.generated.pt'))
    
    raise Exception(f'Unknown artifact: {ref}')

def store_artifact(ref: str, type: str, file: str):
    artifact = wandb.Artifact(ref, type=type)
    artifact.add_file(file)
    wandb.log_artifact(artifact)
