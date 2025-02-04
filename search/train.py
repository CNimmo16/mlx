import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing
import torch
import tqdm
from typing import TypedDict
import swifter
import numpy as np
import os
import wandb
import sklearn

from models import two_towers
from util import devices, artifacts, mini, constants

dirname = os.path.dirname(__file__)

from models import doc_embedder, query_embedder, two_towers, vectors

BATCH_SIZE = 64

EPOCHS = 100

torch.manual_seed(16)

class Doc(TypedDict):
    doc_text: str
    doc_ref: str

class TrainingRow(TypedDict):
    query: str
    relevant_doc: Doc
    non_relevant_doc: Doc

def calc_distance(query, document):
    return 1 - torch.nn.functional.cosine_similarity(query, document)

def get_irrelevant_doc_embedding(row, all_rows):
    for i in range(0, 100):
        other_row = all_rows.sample(1).iloc[0]
        if (other_row['query'] != row['query']) and (other_row['doc_ref'] != row['doc_ref']):
            return doc_embedder.get_embeddings_for_doc(other_row['doc_text'])
    raise ValueError("No non-relevant result found in random 100 row sample. Weird.")

def train():
    wandb.init(project='search', name='search-mini' if mini.is_mini() else 'search')
    wandb.config = {
        "learning_rate": two_towers.LEARNING_RATE,
        "query_hidden_layer_dimensions": two_towers.QUERY_HIDDEN_LAYER_DIMENSIONS,
        "doc_hidden_layer_dimensions": two_towers.DOC_HIDDEN_LAYER_DIMENSIONS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS
    }

    device = devices.get_device()

    data = pd.read_csv(constants.RESULTS_PATH)

    train, val = sklearn.model_selection.train_test_split(data, test_size=0.2, random_state=16)

    model = two_towers.Model().to(device)

    calc_loss = torch.nn.TripletMarginWithDistanceLoss(margin=two_towers.MARGIN, distance_function=calc_distance)
    optimizer = torch.optim.Adam(model.parameters(), lr=two_towers.LEARNING_RATE)

    best_val_loss = float('inf')
    best_state_dict = None
    
    vectors.get_vecs()

    train_batches = np.array_split(train, len(train) // BATCH_SIZE)
    val_batches = np.array_split(val, len(val) // BATCH_SIZE)
    for epoch in range(EPOCHS):
        print('=====')
        train_loss = 0.0
        for batch in tqdm.tqdm(train_batches, desc=f"Epoch {epoch+1}", leave=False):
            query_embeddings = batch['query'].swifter.progress_bar(False).apply(query_embedder.get_embeddings_for_query)
            query_embeddings = torch.tensor(np.vstack(query_embeddings.values)).to(device)

            relevant_doc_embeddings = batch['doc_text'].swifter.progress_bar(False).apply(doc_embedder.get_embeddings_for_doc)
            relevant_doc_embeddings = torch.tensor(np.vstack(relevant_doc_embeddings.values)).to(device)

            irrelevant_doc_embeddings = batch.apply(lambda row: get_irrelevant_doc_embedding(row, train), axis=1)
            irrelevant_doc_embeddings = torch.tensor(np.vstack(irrelevant_doc_embeddings.values)).to(device)

            optimizer.zero_grad()

            query_outputs = model.encode_query(query_embeddings)
            relevant_doc_outputs = model.encode_doc(relevant_doc_embeddings)
            irrelevant_doc_outputs = model.encode_doc(irrelevant_doc_embeddings)

            loss = calc_loss(query_outputs, relevant_doc_outputs, irrelevant_doc_outputs)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print('Train loss', train_loss / len(train))
        print('Calculating val loss...')

        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_batches:
                query_embeddings = batch['query'].swifter.progress_bar(False).apply(query_embedder.get_embeddings_for_query)
                query_embeddings = torch.tensor(np.vstack(query_embeddings.values)).to(device)

                relevant_doc_embeddings = batch['doc_text'].swifter.progress_bar(False).apply(doc_embedder.get_embeddings_for_doc)
                relevant_doc_embeddings = torch.tensor(np.vstack(relevant_doc_embeddings.values)).to(device)

                irrelevant_doc_embeddings = batch.apply(lambda row: get_irrelevant_doc_embedding(row, train), axis=1)
                irrelevant_doc_embeddings = torch.tensor(np.vstack(irrelevant_doc_embeddings.values)).to(device)

                query_outputs = model.encode_query(query_embeddings)
                relevant_doc_outputs = model.encode_doc(relevant_doc_embeddings)
                irrelevant_doc_outputs = model.encode_doc(irrelevant_doc_embeddings)

                loss = calc_loss(query_outputs, relevant_doc_outputs, irrelevant_doc_outputs)

                val_loss += loss.item()

        print('Val loss', val_loss / len(val))

        wandb.log({ 'epoch': epoch + 1, 'train-loss': train_loss, 'val_loss': val_loss })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()

    model_save_path = os.path.join(dirname, 'data/two-towers-weights.generated.pt')
    torch.save(best_state_dict, model_save_path)
    artifacts.store_artifact('two-towers-weights', 'model', model_save_path)

    wandb.finish()

train()
