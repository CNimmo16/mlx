import pandas as pd
import nltk
from models import skipgram
import string

nltk.download('stopwords')

def remove_punctuation(token: str) -> str:
    return token.translate(str.maketrans("", "", string.punctuation))

def trim(token: str) -> str:
    return token.strip()

def lowercase(token: str) -> str:
    return token.lower()

def stem(token: str) -> str:
    if (token.endswith("ing")):
        return token[:-3]
    return token

pipeline = [
    lowercase,
    remove_punctuation,
    stem,
    trim
]
def run_pipeline(text: str):
    ret = text
    for fn in pipeline:
        ret = fn(ret)
    return ret

def build_vocab(tokens: pd.DataFrame):
    vocab = tokens['token'].dropna().apply(run_pipeline)

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

def tokenize(vocab, text):
    processed_tokens = [run_pipeline(token) for token in text.lower().split()]
    return [token if token in vocab.index else 'unk' for token in processed_tokens]

def getIdFromToken(vocab: pd.DataFrame, token: str):
    try:
        return int(vocab.at[token, 'id'])
    except:
        return int(vocab.at['unk', 'id'])

def getTokenFromId(vocab: pd.DataFrame, id: float):
    return vocab[vocab['id'] == id].index.values[0]
