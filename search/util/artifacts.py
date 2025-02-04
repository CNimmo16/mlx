import torch
import pandas as pd
import os
import wandb

import os
dirname = os.path.dirname(__file__)

api = wandb.Api()

def download_from_wandb(ref: str, file: str):
    artifact = api.artifact(f"cnimmo16/search/{ref}:latest")
    dir = artifact.download(os.path.join(dirname, '../artifacts'))
    return os.path.join(dir, file)
    
def load_artifact(ref: str):
    if (ref == 'two-towers-weights'):
        predictor_weights_path = download_from_wandb('two-towers-weights', 'two-towers-weights.generated.pt')
        return torch.load(predictor_weights_path, map_location=torch.device('cpu'))
    
    raise Exception(f'Unknown artifact: {ref}')

def store_artifact(ref: str, type: str, file: str):
    artifact = wandb.Artifact(ref, type=type)
    artifact.add_file(file)
    wandb.log_artifact(artifact)
