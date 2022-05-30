from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import numpy as np
import torch
from fastai.vision.all import load_learner, Image
import torchvision.transforms as T


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

learn = load_learner("models/model_99.pkl")
learn.precompute = False


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(image: UploadFile = File(...)):

    # process image
    img = Image.open(image.file).resize((28, 28))
    img = (T.ToTensor()(img)[3] * 255).to(torch.uint8)

    # predict
    pred = learn.predict(img)

    return {"label": pred[0]}
