from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Annotated
import prediction

import os
dirname = os.path.dirname(__file__)

app = FastAPI()

templates = Jinja2Templates(directory=os.path.join(dirname, "web/templates"))

static_dir = os.path.join(dirname, "web/static")

app.mount(static_dir, StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.j2"
    )

@app.post("/predict", response_class=HTMLResponse)
async def root(request: Request, post_title: Annotated[str, Form()], karma: Annotated[float, Form()]):
    try:
        result = prediction.predict(post_title, karma)
        return templates.TemplateResponse(
            request=request, name="index.j2", context={"prediction": result, "post_title": post_title, "karma": karma}
        )
    except:
        return templates.TemplateResponse(
            request=request, name="index.j2", context={"error": "Something went wrong - no recognised tokens in provided title", "post_title": post_title, "karma": karma}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
