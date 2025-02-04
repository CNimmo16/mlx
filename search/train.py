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

import models
import models.doc_projector, models.query_projector, models.doc_embedder, models.query_embedder, models.vectors
from util import devices, artifacts, mini, constants

dirname = os.path.dirname(__file__)

BATCH_SIZE = 64

EPOCHS = 100
LEARNING_RATE = 0.0002
MARGIN = 0.2

torch.manual_seed(16)

def calc_distance(query, document):
    return 1 - torch.nn.functional.cosine_similarity(query, document)

def get_irrelevant_doc_embedding(row, all_rows):
    for i in range(0, 100):
        other_row = all_rows.sample(1).iloc[0]
        if (other_row['query'] != row['query']) and (other_row['doc_ref'] != row['doc_ref']):
            return models.doc_embedder.get_embeddings_for_doc(other_row['doc_text'])
    raise ValueError("No non-relevant result found in random 100 row sample. Weird.")

def train():
    wandb.init(project='search', name='search-mini' if mini.is_mini() else 'search')
    wandb.config = {
        "learning_rate": LEARNING_RATE,
        "query_hidden_layer_dimensions": models.query_projector.QUERY_HIDDEN_LAYER_DIMENSIONS,
        "doc_hidden_layer_dimensions": models.doc_projector.DOC_HIDDEN_LAYER_DIMENSIONS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS
    }

    device = devices.get_device()

    data = pd.read_csv(constants.RESULTS_PATH)

    train, val = sklearn.model_selection.train_test_split(data, test_size=0.2, random_state=16)

    query_projector = models.query_projector.Model().to(device)
    doc_projector = models.doc_projector.Model().to(device)

    calc_loss = torch.nn.TripletMarginWithDistanceLoss(margin=MARGIN, distance_function=calc_distance)

    all_params = list(query_projector.parameters()) + list(doc_projector.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=LEARNING_RATE)

    best_val_loss = float('inf')
    best_query_state_dict = None
    best_doc_state_dict = None
    
    models.vectors.get_vecs()

    train_batches = np.array_split(train, len(train) // BATCH_SIZE)
    val_batches = np.array_split(val, len(val) // BATCH_SIZE)
    for epoch in range(EPOCHS):
        print('=====')
        train_loss = 0.0
        for batch in tqdm.tqdm(train_batches, desc=f"Epoch {epoch+1}", leave=False):
            query_embeddings = batch['query'].swifter.progress_bar(False).apply(models.query_embedder.get_embeddings_for_query)
            query_embeddings = torch.tensor(np.vstack(query_embeddings.values)).to(device)

            relevant_doc_embeddings = batch['doc_text'].swifter.progress_bar(False).apply(models.doc_embedder.get_embeddings_for_doc)
            relevant_doc_embeddings = torch.tensor(np.vstack(relevant_doc_embeddings.values)).to(device)

            irrelevant_doc_embeddings = batch.apply(lambda row: get_irrelevant_doc_embedding(row, train), axis=1)
            irrelevant_doc_embeddings = torch.tensor(np.vstack(irrelevant_doc_embeddings.values)).to(device)

            optimizer.zero_grad()

            query_outputs = query_projector(query_embeddings)
            relevant_doc_outputs = doc_projector(relevant_doc_embeddings)
            irrelevant_doc_outputs = doc_projector(irrelevant_doc_embeddings)

            loss = calc_loss(query_outputs, relevant_doc_outputs, irrelevant_doc_outputs)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print('Train loss', train_loss / len(train))
        print('Calculating val loss...')

        query_projector.eval()
        doc_projector.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_batches:
                query_embeddings = batch['query'].swifter.progress_bar(False).apply(models.query_embedder.get_embeddings_for_query)
                query_embeddings = torch.tensor(np.vstack(query_embeddings.values)).to(device)

                relevant_doc_embeddings = batch['doc_text'].swifter.progress_bar(False).apply(models.doc_embedder.get_embeddings_for_doc)
                relevant_doc_embeddings = torch.tensor(np.vstack(relevant_doc_embeddings.values)).to(device)

                irrelevant_doc_embeddings = batch.apply(lambda row: get_irrelevant_doc_embedding(row, train), axis=1)
                irrelevant_doc_embeddings = torch.tensor(np.vstack(irrelevant_doc_embeddings.values)).to(device)

                query_outputs = query_projector(query_embeddings)
                relevant_doc_outputs = doc_projector(relevant_doc_embeddings)
                irrelevant_doc_outputs = doc_projector(irrelevant_doc_embeddings)

                loss = calc_loss(query_outputs, relevant_doc_outputs, irrelevant_doc_outputs)

                val_loss += loss.item()

        print('Val loss', val_loss / len(val))

        wandb.log({ 'epoch': epoch + 1, 'train-loss': train_loss, 'val_loss': val_loss })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_query_state_dict = query_projector.state_dict()
            best_doc_state_dict = doc_projector.state_dict()

    query_model_save_path = os.path.join(dirname, 'data/query-projector-weights.generated.pt')
    torch.save(best_query_state_dict, query_model_save_path)
    artifacts.store_artifact('query-projector-weights', 'model', query_model_save_path)

    doc_model_save_path = os.path.join(dirname, 'data/doc-projector-weights.generated.pt')
    torch.save(best_doc_state_dict, doc_model_save_path)
    artifacts.store_artifact('doc-projector-weights', 'model', doc_model_save_path)

    wandb.finish()

train()
