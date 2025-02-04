import os
import sys

dirname = os.path.dirname(__file__)

src_path = os.path.abspath(os.path.join(dirname, '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    
import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing
import torch
import tqdm
from typing import TypedDict
import swifter
import numpy as np
import wandb
import sklearn

import models
import models.doc_projector, models.query_projector, models.doc_embedder, models.query_embedder, models.vectors
import dataset
from util import devices, artifacts, mini, constants

BATCH_SIZE = 64

EPOCHS = 100
LEARNING_RATE = 0.0002
MARGIN = 0.2

torch.manual_seed(16)

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

    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)

    train_loader = torch.utils.data.DataLoader(dataset.TwoTowerDataset(train), batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.collate_two_tower_batch)
    val_loader = torch.utils.data.DataLoader(dataset.TwoTowerDataset(val), batch_size=BATCH_SIZE, shuffle=False, collate_fn=dataset.collate_two_tower_batch)

    query_projector = models.query_projector.Model().to(device)
    doc_projector = models.doc_projector.Model().to(device)

    calc_loss = torch.nn.TripletMarginWithDistanceLoss(margin=MARGIN, distance_function=lambda query, doc: 1 - torch.nn.functional.cosine_similarity(query, doc))

    all_params = list(query_projector.parameters()) + list(doc_projector.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=LEARNING_RATE)

    best_val_loss = float('inf')
    best_query_state_dict = None
    best_doc_state_dict = None
    
    models.vectors.get_vecs()

    for epoch in range(EPOCHS):
        print('=====')
        train_loss = 0.0
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):

            # query_embeddings = batch['query'].swifter.progress_bar(False).apply(models.query_embedder.get_embeddings_for_query)
            # query_embeddings = torch.tensor(np.vstack(query_embeddings.values)).to(device)

            # relevant_doc_embeddings = batch['doc_text'].swifter.progress_bar(False).apply(models.doc_embedder.get_embeddings_for_doc)
            # relevant_doc_embeddings = torch.tensor(np.vstack(relevant_doc_embeddings.values)).to(device)

            # irrelevant_doc_embeddings = batch.apply(lambda row: get_irrelevant_doc_embedding(row, train), axis=1)
            # irrelevant_doc_embeddings = torch.tensor(np.vstack(irrelevant_doc_embeddings.values)).to(device)

            optimizer.zero_grad()

            query_outputs = query_projector(batch['query_embeddings'])
            relevant_doc_outputs = doc_projector(batch['relevant_doc_embeddings'])
            irrelevant_doc_outputs = doc_projector(batch['irrelevant_doc_embeddings'])

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
            for batch in val_loader:
                query_outputs = query_projector(batch['query_embeddings'])
                relevant_doc_outputs = doc_projector(batch['relevant_doc_embeddings'])
                irrelevant_doc_outputs = doc_projector(batch['irrelevant_doc_embeddings'])

                loss = calc_loss(query_outputs, relevant_doc_outputs, irrelevant_doc_outputs)

                val_loss += loss.item()

        print('Val loss', val_loss / len(val))

        wandb.log({ 'epoch': epoch + 1, 'train-loss': train_loss, 'val_loss': val_loss })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_query_state_dict = query_projector.state_dict()
            best_doc_state_dict = doc_projector.state_dict()

    query_model_save_path = os.path.join(dirname, '../data/query-projector-weights.generated.pt')
    torch.save(best_query_state_dict, query_model_save_path)
    artifacts.store_artifact('query-projector-weights', 'model', query_model_save_path)

    doc_model_save_path = os.path.join(dirname, '../data/doc-projector-weights.generated.pt')
    torch.save(best_doc_state_dict, doc_model_save_path)
    artifacts.store_artifact('doc-projector-weights', 'model', doc_model_save_path)

    wandb.finish()

train()
