from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Annotated
import requests
import json
import random
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

import os
dirname = os.path.dirname(__file__)

app = FastAPI()

templates = Jinja2Templates(directory=os.path.join(dirname, "web/templates"))

static_dir = os.path.join(dirname, "web/static")

app.mount(static_dir, StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        request=request, name="secretindex.j2"
    )

cache = {}

@app.post("/predict", response_class=HTMLResponse)
async def root(request: Request, post_title: Annotated[str, Form()]):
    if (post_title in cache):
        score = cache[post_title]
    else:
        res = requests.get(f"https://hn.algolia.com/api/v1/search?query={post_title}&tags=story")
        response = json.loads(res.text)
        if (len(response['hits']) > 0 and similar(post_title, response['hits'][0]['title']) > 0.9):
            score = response['hits'][0]['points']
            score = random.randint(int(score * 0.75), int(score * 1.25))
        else:
            score = random.randint(1, 100)
        cache[post_title] = score
    return templates.TemplateResponse(
        request=request, name="secretindex.j2", context={"prediction": score, "post_title": post_title}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
