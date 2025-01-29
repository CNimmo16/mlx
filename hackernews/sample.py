import tqdm
import collections
import more_itertools
import requests
import wandb
import torch
from util import cache
import nltk
import pandas as pd
import string

VOCAB_SIZE = 50000

items_table = "hacker_news.items"

torch.manual_seed(16)

print('Loading text8')
r = requests.get("https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8")
with open("text8", "wb") as f: f.write(r.content)
with open('text8') as f: text8: str = f.read()

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
limit = VOCAB_SIZE * 1.1 # 10% more to account for duplicates after stemming etc. (removed later)
common_words = cache.query("words", f"""SELECT
    lower(unnest(string_to_array(regexp_replace(title, '[^a-zA-Z0-9'']', ' ', 'g'), ' '))) AS token,
    count(*) AS count
    FROM {items_table}
    WHERE type = 'story' AND title IS NOT null
    GROUP BY token
    ORDER BY count DESC
    LIMIT {limit}
""")

print(f"Got {len(common_words)} unique words")

## remove the stopwords
common_words = common_words[(~common_words["token"].isin(nltk.corpus.stopwords.words("english"))) & (common_words["token"] != "")].dropna().drop("count", axis=1).reset_index(drop=True)

print(f"Got {len(common_words)} tokens after dropping stopwords and empty tokens")

def runPipeline(text: str):
    ret = text
    for fn in pipeline:
        ret = fn(ret)
    return ret
vocab = common_words['token'].apply(runPipeline)

vocab = vocab.drop_duplicates(keep='first')[:VOCAB_SIZE]

print(f"Got {len(vocab)} tokens after truncating to {VOCAB_SIZE}")

## add unknown token
vocab = pd.concat([vocab, pd.Series(['unk'])]).reset_index(drop=True)

vocab = pd.DataFrame({'token': vocab, 'id': range(len(vocab))})

vocab['id'] += 1

vocab.set_index('token', inplace=True)

def getTokenId(token: str):
    try:
        return int(vocab.at[token, 'id'])
    except:
        return int(vocab.at['unk', 'id'])
    
token_ids = [getTokenId(word) for word in vocab]

class SkipGramFoo(torch.nn.Module):
  def __init__(self, voc, emb, _):
    super().__init__()
    self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
    self.ffw = torch.nn.Linear(in_features=emb, out_features=voc, bias=False)
    self.sig = torch.nn.Sigmoid()

  def forward(self, inpt, trgs, rand):
    emb = self.emb(inpt)
    ctx = self.ffw.weight[trgs]
    rnd = self.ffw.weight[rand]
    out = torch.bmm(ctx, emb.unsqueeze(-1)).squeeze()
    rnd = torch.bmm(rnd, emb.unsqueeze(-1)).squeeze()
    out = self.sig(out)
    rnd = self.sig(rnd)
    pst = -out.log().mean()
    ngt = -(1 - rnd + 10**(-3)).log().mean()
    return pst + ngt


#
#
#
args = (len(token_ids), 64, 2)
mFoo = SkipGramFoo(*args)
print('mFoo', sum(p.numel() for p in mFoo.parameters()))
opFoo = torch.optim.Adam(mFoo.parameters(), lr=0.003)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#
#
#
windows = list(more_itertools.windowed(token_ids, 3))
inputs = [w[1] for w in windows]
targets = [[w[0], w[2]] for w in windows]
input_tensor = torch.LongTensor(inputs)
target_tensor = torch.LongTensor(targets)
dataset = torch.utils.data.TensorDataset(input_tensor, target_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)


#
#
#
wandb.init(project='mlx6-word2vec', name='mFoo')
mFoo.to(device)
for epoch in range(1):
  prgs = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
  for inpt, trgs in prgs:
    inpt, trgs = inpt.to(device), trgs.to(device)
    rand = torch.randint(0, len(token_ids), (inpt.size(0), 2)).to(device)
    opFoo.zero_grad()
    loss = mFoo(inpt, trgs, rand)
    loss.backward()
    opFoo.step()
    wandb.log({'loss': loss.item()})


#
#
#
print('Saving...')
torch.save(mFoo.state_dict(), './weights.pt')
print('Uploading...')
artifact = wandb.Artifact('model-weights', type='model')
artifact.add_file('./weights.pt')
wandb.log_artifact(artifact)
print('Done!')
wandb.finish()
