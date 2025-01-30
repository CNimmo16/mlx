import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import more_itertools
import pandas as pd
from util import cache, artifacts
import string
import nltk
from ast import literal_eval
import tqdm
import math
import wandb
from random import sample
from models import skipgram
import tokenization
import swifter # do not remove - used indirectly by DataFrame.swifter

import importlib
importlib.reload(cache)

import os
dirname = os.path.dirname(__file__)

nltk.download('stopwords')

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

items_table = "hacker_news.items"

# Tokenization

print("Fetching unique tokens from database...")

## take the most common tokens
if skipgram.MINIMODE:
    mlDef = "Machine learning (ML) is a branch of artificial intelligence (AI) that enables computers to learn from data and make decisions or predictions without being explicitly programmed. Instead of following rigid rules, machine learning models recognize patterns in data and improve their performance over time."
    tokens = pd.DataFrame({ 'token': mlDef.split(' '), 'count': [1 for x in mlDef.split(' ')] })
else:
    limit = skipgram.MAX_VOCAB_SIZE * 1.1 # 10% more to account for duplicates after stemming etc. (removed later)
    tokens = cache.query("words", f"""SELECT
        lower(unnest(string_to_array(regexp_replace(title, '[^a-zA-Z0-9'']', ' ', 'g'), ' '))) AS token,
        count(*) AS count
        FROM {items_table}
        WHERE type = 'story' AND title IS NOT null
        GROUP BY token
        ORDER BY count DESC
        LIMIT {limit}
    """)

print(f"Got {len(tokens)} unique tokens")

print(f"Got {len(tokens)} tokens after dropping stopwords and empty tokens")

vocab = tokens['token'].dropna().apply(tokenization.run_pipeline)

vocab = vocab.drop_duplicates(keep='first')

## remove the stopwords
vocab = vocab[(~vocab.isin(nltk.corpus.stopwords.words("english"))) & (vocab != "")].dropna().reset_index(drop=True)

vocab = vocab[:skipgram.MAX_VOCAB_SIZE]

print(f"Got {len(vocab)} tokens after truncating to {skipgram.MAX_VOCAB_SIZE}")

## add unknown token
vocab = pd.concat([vocab, pd.Series(['unk'])]).reset_index(drop=True)

vocab = pd.DataFrame({'token': vocab, 'id': range(len(vocab))})

vocab['id'] += 1

vocab.set_index('token', inplace=True)

def getIdFromToken(token: str):
    try:
        return int(vocab.at[token, 'id'])
    except:
        return int(vocab.at['unk', 'id'])

def getTokenFromId(id: float):
    return vocab[vocab['id'] == id].index.values[0]

vocab.to_csv(os.path.join(dirname, f"data/vocab{'__mini' if skipgram.MINIMODE else ''}.generated.csv"), index=True)

# Generate training data

def create_skipgram_data(tokens, window_size):
    tokens = [token for token in tokens if token != getIdFromToken('unk')]
    targets = []
    contexts = []
    for i in range(len(tokens)):
        target = tokens[i]
        # Get context indices
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        context = [tokens[j] for j in range(start, end) if j != i]
        # Create training pairs
        for c in context:
            targets.append(target)
            contexts.append(c)
    return targets, contexts

def processText(text: str):
    tokens = tokenization.tokenize(text)
    token_ids = [getIdFromToken(word) for word in tokens]
    return create_skipgram_data(token_ids, skipgram.WINDOW_SIZE)

# Get skipgram data from wikipedia articles

print("====")

if skipgram.MINIMODE:
    targets, contexts = processText(mlDef)

    processed = pd.DataFrame({ 'target': targets, 'context': contexts })

    processed.dropna(inplace=True)

    final_skipgram_data = list(processed.itertuples(index=False, name=None))
else:
    wiki_data_path = os.path.join(dirname, f"data/wiki_skipgram_data.generated.csv")
    try:
        wiki_data = pd.read_csv(wiki_data_path)
        print("CACHE HIT: Got existing wiki skipgram data from file...")
    except:
        print("Loading wiki articles from file...")

        def load_articles():
            ret = pd.read_xml(os.path.join(dirname, 'data/enwik8.xml'), xpath='//pages/page/revision/text')

            return ret[ret['text'].str.match(r'^( )*#redirect', case=False) == False].reset_index()

        wiki_articles = cache.frame(
            ref='wiki_articles',
            version="0",
            getFrame=load_articles,
        )

        if skipgram.MINIMODE:
            wiki_articles = wiki_articles[:10]

        chunk_size = 1000
        header = True
        prgs = tqdm.tqdm(np.array_split(wiki_articles, math.ceil(len(wiki_articles) / chunk_size)),
                            desc=f"Building skipgram data for {len(wiki_articles)} wiki articles", leave=False)
        for chunk in prgs:
            targets_and_contexts = chunk['text'].swifter.progress_bar(False).apply(processText)

            targets = [d[0] for d in targets_and_contexts]
            contexts = [d[1] for d in targets_and_contexts]

            flat_targets = [val for sublist in targets for val in sublist]
            flat_contexts = [val for sublist in contexts for val in sublist]

            processed = pd.DataFrame({ 'target': flat_targets, 'context': flat_contexts })
            
            processed.dropna(inplace=True)

            processed.to_csv(wiki_data_path, header=header, mode='a', index=False)

            header = False
        wiki_data = pd.read_csv(wiki_data_path)

    print("> transforming to list...")
    wiki_skipgram_data = list(wiki_data.itertuples(index=False, name=None))

    print(f"Generated {len(wiki_skipgram_data)} skipgram data points from wiki articles")

    wiki_data_size = int(skipgram.TRAINING_DATA_SIZE * (1 - skipgram.HACKER_NEWS_RATIO))
    print(f"Truncating to {wiki_data_size} data points...")
    wiki_skipgram_data = wiki_skipgram_data[:wiki_data_size]

    ## Get skipgram data from hacker news posts

    print("====")

    hn_data_path = os.path.join(dirname, f"data/hn_skipgram_data.generated.csv")
    try:
        hn_data = pd.read_csv(os.path.join(dirname, hn_data_path))
        print("CACHE HIT: Got existing hacker news skipgram data from file...")
    except:
        print("Loading hacker news posts from db...")

        hn_posts = cache.query("titles", f"""SELECT
            title
            FROM {items_table}
            WHERE type = 'story' AND title IS NOT null
            LIMIT 1000000
        """)[['title']].rename(columns={'title': 'text'})

        hn_posts.dropna(inplace=True)

        chunk_size = 1000
        header = True
        prgs = tqdm.tqdm(np.array_split(hn_posts, math.ceil(len(hn_posts) / chunk_size)),
                            desc=f"Building skipgram data for {len(hn_posts)} hacker news posts", leave=False)
        for chunk in prgs:
            targets_and_contexts = chunk['text'].swifter.progress_bar(False).apply(processText)

            targets = [d[0] for d in targets_and_contexts]
            contexts = [d[1] for d in targets_and_contexts]

            flat_targets = [val for sublist in targets for val in sublist]
            flat_contexts = [val for sublist in contexts for val in sublist]

            processed = pd.DataFrame({ 'target': flat_targets, 'context': flat_contexts })
            
            processed.dropna(inplace=True)

            processed.to_csv(hn_data_path, header=header, mode='a', index=False)

            header = False
        hn_data = pd.read_csv(hn_data_path)

    print("> transforming to list...")
    hn_skipgram_data = list(hn_data.itertuples(index=False, name=None))

    hn_data_size = int(skipgram.TRAINING_DATA_SIZE * skipgram.HACKER_NEWS_RATIO)
    print(f"Truncating to {hn_data_size} data points...")
    hn_skipgram_data = hn_skipgram_data[:hn_data_size]

    print(f"Generated {len(hn_skipgram_data)} skipgram data points from hacker news posts")

    final_skipgram_data = wiki_skipgram_data + hn_skipgram_data

    print(f"Concatenated skipgram data resulting in {len(final_skipgram_data)} total datapoints")

# Train model
wandb.init(project='word2vec', name='skipgram-mini' if skipgram.MINIMODE else 'skipgram')
wandb.config = {
    "learning_rate": skipgram.LEARNING_RATE,
    "embedding_dim": skipgram.EMBEDDING_DIM,
    "batch_size": skipgram.BATCH_SIZE,
    "epochs": skipgram.EPOCHS
}

# Dataset and DataLoader
targets = [w[0] for w in final_skipgram_data] # all the "targets"
contexts = [w[1] for w in final_skipgram_data] # all the "contexts"
input_tensor = torch.LongTensor(targets)
expected_output_tensor = torch.LongTensor(contexts)

dataset = torch.utils.data.TensorDataset(input_tensor, expected_output_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10 if skipgram.MINIMODE else 512, shuffle=True)

# Initialize model, loss, and optimizer
model = skipgram.Model(vocab.size + 1, skipgram.EMBEDDING_DIM)

model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=skipgram.LEARNING_RATE)

if skipgram.MINIMODE:
    print(f"Inputs:")
    print([(getTokenFromId(x[0]), getTokenFromId(x[1])) for x in final_skipgram_data[:10]], sep='\n')

# Training loop
for epoch in range(skipgram.EPOCHS):
    batches = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
    for batch in batches:
        targets, contexts = batch
        targets, contexts = targets.to(device), contexts.to(device)

        optimizer.zero_grad()
        model_output_logits = model(targets)
        loss = criterion(model_output_logits, contexts)
        loss.backward()
        optimizer.step()
        wandb.log({'epoch': epoch + 1, 'train-loss': loss.item()})

# Save model
print('Saving...')
artifacts.save_artifact(model.state_dict(), 'model-weights', 'model', os.path.join(dirname, 'data/weights.generated.pt'))

pd.DataFrame(model.embeddings.weight.data.cpu()).to_csv(os.path.join(dirname, 'data/embeddings.generated.csv'), index=False)
artifacts.save_artifact(None, 'embeddings', 'dataset', os.path.join(dirname, 'data/embeddings.generated.csv'))

artifacts.save_artifact(None, 'vocab', 'dataset', os.path.join(dirname, 'data/vocab.generated.csv'))

print('Done!')
wandb.finish()
