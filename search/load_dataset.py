import datasets as hf_datasets
import pandas as pd
import os
import swifter

dirname = os.path.dirname(__file__)

def expand_passages(row) -> hf_datasets.Dataset:
    queries = []
    doc_refs = []
    doc_texts = []
    is_selecteds = []
    for i, is_selected in enumerate(row['passages']['is_selected']):
        queries.append(row['query'])
        doc_refs.append(row['passages']['url'][i])
        doc_texts.append(row['passages']['passage_text'][i])
        is_selecteds.append(is_selected)

    return pd.DataFrame({
        'query': queries,
        'doc_ref': doc_refs,
        'doc_text': doc_texts,
        'is_selected': is_selecteds
    })

def run():
    print('Loading dataset...')

    df = hf_datasets.load_dataset("microsoft/ms_marco", "v1.1", split="train").to_pandas()

    print('Expanding passages...')

    df = pd.concat(df.swifter.apply(expand_passages, axis=1).tolist(), ignore_index=True)

    print('Writing to file...')

    df.head()

    df.to_csv(os.path.join(dirname, "./data/results.generated.csv"), index=False)
    df[0:1000].to_csv(os.path.join(dirname, "./data/results-mini.generated.csv"), index=False)

    print('Done!')

run()
