import torch
from models import skipgram
import pandas as pd
import os
from util import artifacts, cache

items_table = "hacker_news.items"

loaded_artifacts = artifacts.load_artifacts()

embeddings = loaded_artifacts['state_dict']
vocab = loaded_artifacts['vocab']

print(vocab)

model = skipgram.Model(vocab.size + 1, skipgram.EMBEDDING_DIM)

model.load_state_dict(embeddings)
model.eval()

# hn_posts = cache.query("titles", f"""SELECT
#     title,
#     score
#     FROM {items_table}
#     WHERE type = 'story' AND title IS NOT null
#     LIMIT 1000000
# """)

# def get_embeddings(title):
#     tokens = skipgram.tokenize(title)

# hn_posts['embeddings'] = hn_posts['title'].apply(get_embeddings)
