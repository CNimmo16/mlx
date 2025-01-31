from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Annotated
import random

import os
dirname = os.path.dirname(__file__)

app = FastAPI()

templates = Jinja2Templates(directory=os.path.join(dirname, "templates"))

static_dir = os.path.join(dirname, "static")

app.mount(static_dir, StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.j2"
    )

@app.post("/predict", response_class=HTMLResponse)
async def root(request: Request, post_title: Annotated[str, Form()], username: Annotated[str, Form()]):
    prediction = random.randint(0, 100)
    return templates.TemplateResponse(
        request=request, name="index.j2", context={"prediction": prediction, "post_title": post_title, "username": username}
    )
