import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from util import cache
import string
import nltk
from ast import literal_eval
import tqdm
import math
import wandb
from random import sample
from models import skipgram
import swifter # do not remove - used indirectly by DataFrame.swifter

import importlib
importlib.reload(cache)

import os
dirname = os.path.dirname(__file__)

torch.manual_seed(42)

MINI=True

items_table = "hacker_news.items"

# Tokenization

def removePunctuation(token: str) -> str:
    return token.translate(str.maketrans("", "", string.punctuation))

def trim(token: str) -> str:
    return token.strip()

def lowercase(token: str) -> str:
    return token.lower()

def stemIng(token: str) -> str:
    if (token.endswith("ing")):
        return token[:-3]
    return token

pipeline = [
    lowercase,
    removePunctuation,
    stemIng,
    trim
]

print("Fetching unique tokens from database...")

## take the most common tokens
limit = skipgram.VOCAB_SIZE * 1.1 # 10% more to account for duplicates after stemming etc. (removed later)
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

## remove the stopwords
tokens = tokens[(~tokens["token"].isin(nltk.corpus.stopwords.words("english"))) & (tokens["token"] != "")].dropna().drop("count", axis=1).reset_index(drop=True)

print(f"Got {len(tokens)} tokens after dropping stopwords and empty tokens")

def runPipeline(text: str):
    ret = text
    for fn in pipeline:
        ret = fn(ret)
    return ret
vocab = tokens['token'].apply(runPipeline)

vocab = vocab.drop_duplicates(keep='first')[:skipgram.VOCAB_SIZE]

print(f"Got {len(vocab)} tokens after truncating to {skipgram.VOCAB_SIZE}")

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
    return vocab.at[id, 'token']

vocab.to_csv(os.path.join(dirname, 'data/vocab.generated.csv'), index=True)

# Generate training data

def tokenize(text):
    processed_tokens = [runPipeline(token) for token in text.lower().split()]
    return [token if token in vocab.index else 'unk' for token in processed_tokens]

def create_skipgram_data(tokens, window_size):
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
    tokens = tokenize(text)
    token_ids = [getIdFromToken(word) for word in tokens]
    return create_skipgram_data(token_ids, skipgram.WINDOW_SIZE)

# Get skipgram data from wikipedia articles

print("====")

try:
    wiki_data = pd.read_csv(os.path.join(dirname, 'data/wiki_skipgram_data.generated.csv'))
    print("CACHE HIT: Got existing wiki skipgram data from file...")
except:
    print("Loading wiki articles from file...")

    def load_articles():
        ret = pd.read_xml(os.path.join(dirname, 'data/enwik8.xml', xpath='//pages/page/revision/text'))

        return ret[ret['text'].str.match(r'^( )*#redirect', case=False) == False].reset_index()

    wiki_articles = cache.frame(
        ref='wiki_articles',
        version="0",
        getFrame=load_articles,
    )

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

        processed.to_csv(os.path.join(dirname, 'data/wiki_skipgram_data.generated.csv'), header=header, mode='a', index=False)

        header = False
    wiki_data = pd.read_csv(os.path.join(dirname, 'data/wiki_skipgram_data.generated.csv'))

print("> transforming to list...")
wiki_skipgram_data = list(wiki_data.itertuples(index=False, name=None))

print(f"Generated {len(wiki_skipgram_data)} skipgram data points from wiki articles")

wiki_data_size = int(skipgram.TRAINING_DATA_SIZE * (1 - skipgram.HACKER_NEWS_RATIO))
print(f"Truncating to {wiki_data_size} data points...")
wiki_skipgram_data = wiki_skipgram_data[:wiki_data_size]

## Get skipgram data from hacker news posts

print("====")

try:
    hn_data = pd.read_csv(os.path.join(dirname, 'data/hn_skipgram_data.generated.csv'))
    print("CACHE HIT: Got existing hacker news skipgram data from file...")
except:
    print("Loading hacker news posts from db...")

    hn_posts = cache.query("titles", f"""SELECT
        title
        FROM {items_table}
        WHERE type = 'story' AND title IS NOT null
        LIMIT 100000
    """)[['title']].rename(columns={'title': 'text'})

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

        processed.to_csv(os.path.join(dirname, 'data/hn_skipgram_data.generated.csv'), header=header, mode='a', index=False)

        header = False
    hn_data = pd.read_csv(os.path.join(dirname, 'data/hn_skipgram_data.generated.csv'))

print("> transforming to list...")
hn_skipgram_data = list(hn_data.itertuples(index=False, name=None))

hn_data_size = int(skipgram.TRAINING_DATA_SIZE * skipgram.HACKER_NEWS_RATIO)
print(f"Truncating to {hn_data_size} data points...")
hn_skipgram_data = hn_skipgram_data[:hn_data_size]

print(f"Generated {len(hn_skipgram_data)} skipgram data points from hacker news posts")

final_skipgram_data = wiki_skipgram_data + hn_skipgram_data

print(f"Concatenated skipgram data resulting in {len(final_skipgram_data)} datapoints")

if MINI:
    MINI_MODE_SAMPLE_SIZE = 100
    final_skipgram_data = sample(final_skipgram_data, MINI_MODE_SAMPLE_SIZE)
    print(f"MINI MODE: sampled {MINI_MODE_SAMPLE_SIZE} data points for final training data")


# Train model
wandb.init(project='word2vec', name='model1')

# Dataset and DataLoader
class SkipGramDataset(Dataset):
    c = 0
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        target = self.data[idx][0]
        context = self.data[idx][1]
        self.c = self.c+1
        return torch.tensor(target, dtype=torch.long), torch.tensor(context, dtype=torch.long)

dataset = SkipGramDataset(final_skipgram_data)
dataloader = DataLoader(dataset, batch_size=skipgram.BATCH_SIZE, shuffle=True)

# Initialize model, loss, and optimizer
model = skipgram.Model(vocab.size + 1, skipgram.EMBEDDING_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=skipgram.LEARNING_RATE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
for epoch in range(skipgram.EPOCHS):
    total_loss = 0
    prgs = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
    for targets, contexts in prgs:
        optimizer.zero_grad()
        outputs = model(targets)
        loss = criterion(outputs, contexts)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        wandb.log({'loss': loss.item()})
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')

# Get embeddings
embeddings = model.embeddings.weight.data

# Save model
print('Saving...')
torch.save(model.state_dict(), os.path.join(dirname, 'data/weights.generated.pt'))
print('Uploading...')
artifact = wandb.Artifact('model-weights', type='model')
artifact.add_file(os.path.join(dirname, 'data/weights.generated.pt'))
wandb.log_artifact(artifact)
print('Done!')
wandb.finish()
