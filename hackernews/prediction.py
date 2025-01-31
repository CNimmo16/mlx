import torch
from util import artifacts
import embeddings
from models import upvote_predictor, skipgram

state_dict = artifacts.load_artifact('predictor-state')
vocab = artifacts.load_artifact('vocab')
embed_scaler = artifacts.load_artifact('predictor-embed-scaler')
karma_scaler = artifacts.load_artifact('predictor-karma-scaler')

model = upvote_predictor.Model(skipgram.EMBEDDING_DIM)

model.load_state_dict(state_dict)
model.eval()

def predict(title, karma):
    title_embeddings = embeddings.get_embeddings_for_title(title)

    scaled_title_embeddings = embed_scaler.transform(title_embeddings.reshape(1, -1))

    embeddings_tensor = torch.tensor(scaled_title_embeddings, dtype=torch.float32)
    
    scaled_karma = karma_scaler.transform([[karma]])

    karma_tensor = torch.tensor(scaled_karma, dtype=torch.float32)

    return model(embeddings_tensor, karma_tensor).detach().numpy()[0]
