import os
import sys

dirname = os.path.dirname(__file__)

src_path = os.path.abspath(os.path.join(dirname, '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import models
import models.query_embedder, models.query_projector, models.vectors
from search import search

def cli():
    models.vectors.get_vecs()

    query = input("Enter a query (or blank to quit): ")

    if not query:
        print('Goodbye!')
        return

    results = search(query)

    for result in results:
        print(f"- {result}")

    cli()

cli()
