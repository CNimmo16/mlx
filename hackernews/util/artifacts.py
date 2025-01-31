import torch
from models import skipgram
import pandas as pd
import os
import wandb

import os
dirname = os.path.dirname(__file__)

api = wandb.Api()

def download_from_wandb(ref: str, file: str):
    artifact = api.artifact(f"cnimmo16/word2vec/{ref}:latest")
    dir = artifact.download(os.path.join(dirname, '../artifacts'))
    return os.path.join(dir, file)

def load_artifacts():
    return {
        'vocab': pd.read_csv(download_from_wandb('vocab', 'vocab.generated.csv')),
        'skipgram_state': torch.load(download_from_wandb('skipgram-weights', 'weights.generated.pt'), map_location=torch.device('cpu')),
        'predictor_state': torch.load(download_from_wandb('predictor-weights', 'weights.generated.pt'), map_location=torch.device('cpu')),
    }

def save_artifact(data, ref: str, type: str, file: str):
    if data is not None:
        torch.save(data, os.path.join(dirname, file))
    artifact = wandb.Artifact(ref, type=type)
    artifact.add_file(file)
    wandb.log_artifact(artifact)
