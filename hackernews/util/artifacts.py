import torch
from models import skipgram
import pandas as pd
import os
import wandb

import os
dirname = os.path.dirname(__file__)

api = wandb.Api()

def download_from_wandb(ref: str, file: str):
    try:
        artifact = api.artifact(f"cnimmo16/word2vec/{ref}:latest")
        dir = artifact.download(os.path.join(dirname, '../artifacts'))
        return os.path.join(dir, file)
    except:
        return None
    
artifacts = None

def load_artifacts():
    # if artifacts:
    #     return artifacts
    vocab_path = download_from_wandb('vocab', 'vocab.generated.csv')
    skipgram_weights_path = download_from_wandb('skipgram-weights', 'skipgram-weights.generated.pt')
    predictor_weights_path = download_from_wandb('predictor-weights', 'predictor-weights.generated.pt')

    return {
        'vocab': pd.read_csv(vocab_path).set_index('token') if vocab_path else None,
        'embeddings': torch.load(skipgram_weights_path, map_location=torch.device('cpu'))['embeddings.weight'] if skipgram_weights_path else None,
        'predictor_state': torch.load(predictor_weights_path, map_location=torch.device('cpu')) if predictor_weights_path else None,
    }

def save_artifact(data, ref: str, type: str, file: str):
    if data is not None:
        torch.save(data, os.path.join(dirname, file))
    artifact = wandb.Artifact(ref, type=type)
    artifact.add_file(file)
    wandb.log_artifact(artifact)
