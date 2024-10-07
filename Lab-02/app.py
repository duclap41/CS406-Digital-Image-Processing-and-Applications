from fastapi import FastAPI, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from typing import Optional

import shutil
import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


INFOS = np.load("static/database/infos.npy")
FEATURES = np.load("static/database/features.npy")
MAPPING = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def get_hist(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist

def search_engine(im_query_path, infos, features, top_k=10, compare_method = "eculidean_distance"):
    hist_query = get_hist(im_query_path).flatten().astype(np.int32)
    # calc distance
    if compare_method == "cosine_similarity":
        similarities = cosine_similarity([hist_query], features).flatten()
        top_k_idx = similarities.argsort()[-top_k:][::-1]

    elif compare_method == "eculidean_distance":
        similarities = np.sqrt(np.sum((features - [hist_query]) ** 2, axis=1))
        top_k_idx = similarities.argsort()[:top_k]

    else:
        raise Exception("The method must 'cosine_similarity' or 'eculidean_distance'!")

    # get top k
    top_k_im = infos[top_k_idx].copy()
    top_k_im[:, 1] = list(map(lambda x: MAPPING[int(x)], top_k_im[:, 1]))
    return top_k_im

app = FastAPI()

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")
os.makedirs("static/uploads", exist_ok=True)

# Set up templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "filename": None})

@app.post("/upload/")
async def upload_image(request: Request, file: UploadFile = File(...)):
    file_path = f"static/uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return templates.TemplateResponse("index.html", {"request": request, "filename": file.filename})

@app.get("/retrieve/")
async def retrieve_image(request: Request, filename: str, method: Optional[str] = None):
    query_im_path = f"static/uploads/{filename}"
    error = False
    try:
        search_results = search_engine(query_im_path, INFOS, FEATURES, top_k=10, compare_method=method)
    except Exception:
        error = True
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "filename": filename,
        "search_results": search_results,
        "selected_method": method,
        "error": error})
