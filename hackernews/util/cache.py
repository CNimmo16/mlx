import pandas as pd
import json
from typing import Callable

pg_url = "postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"

def frame(ref: str, version: str, getFrame: Callable):
    cache_versions_loc = "./data/cache/versions.json"
    try:
        cache_versions_readable = open(cache_versions_loc)
    except FileNotFoundError:
        with open(cache_versions_loc, "w") as f:
            f.write("{}")
        cache_versions_readable = open(cache_versions_loc)
    with cache_versions_readable:
        cache_versions = json.load(cache_versions_readable)
        try:
            df = pd.read_csv(f"./data/cache/{ref}.csv")
            if (ref not in cache_versions or cache_versions[ref] != version):
                raise FileNotFoundError
        except FileNotFoundError:
            df = getFrame()
            df.to_csv(f"./data/cache/{ref}.csv", index=False)
            cache_versions[ref] = version
            with open(cache_versions_loc, 'w', encoding='utf-8') as queries_cache_writable:
                json.dump(cache_versions, queries_cache_writable, ensure_ascii=False, indent=4)
        return df

def query(ref: str, query: str) -> pd.DataFrame:
    return frame(ref, query, lambda: pd.read_sql(query, pg_url))
