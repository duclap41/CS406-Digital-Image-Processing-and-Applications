from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from typing import List

import shutil
import os
import cv2
import numpy as np
import random
import string

app = FastAPI()

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")
os.makedirs("static/uploads", exist_ok=True)

# Set up templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    methods = np.zeros(4)
    return templates.TemplateResponse("index.html", {"request": request, "filename": None, "methods": None})

@app.post("/upload/")
async def upload_image(request: Request, file: UploadFile = File(...)):
    methods = np.zeros(4)
    file_path = f"static/uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return templates.TemplateResponse("index.html", {"request": request, "filename": file.filename, "methods": None})

@app.post("/process_image/")
async def retrieve_image(request: Request, filename: str = Form(...), options: List[str] = Form(...)):
    input_im_path = f"static/uploads/{filename}"
    im_name = gen_random_name() + os.path.splitext(filename)[0]
    image = cv2.imread(input_im_path)
    print(input_im_path)
    methods = np.zeros(4)
    
    if "add_noise" in options:
        im_addnoise = image.copy()
        add_noise(im_addnoise, im_name)
        methods[0] = 1

    if "denoise" in options:
        if methods[0] == 1:
            denoise(im_addnoise, im_name)
        else:
            im_denoise = image.copy()
            denoise(im_denoise, im_name)
        methods[1] = 1
 
    if "sharpen" in options:
        im_sharpen = image.copy()
        sharpen(im_sharpen, im_name)
        methods[2] = 1

    if "edge_detect" in options:
        im_edge_detect = image.copy()
        edge_detect(im_edge_detect, im_name)
        methods[3] = 1

    return templates.TemplateResponse("index.html", {
        "request": request,
        "filename": filename,
        "methods": methods,
        "save_name": im_name,
        })

@app.post("/remove-cache/")
async def remove_cache(request: Request):
    pass

def gen_random_name():
    random_text = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    return random_text
 
# def read_image(image_path):
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     return image

def add_noise(image, name):
    rows, cols,_= image.shape
    noise = np.random.normal(0,15,(rows,cols,3)).astype(np.uint8)
    noisy_image = image + noise
    save_path = f"static/processed/add-noise/{name}.png"
    cv2.imwrite(save_path, noisy_image)

def denoise(image, name):
    # filter noise
    kernel = np.ones((6,6))/36
    image_filtered = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    save_path_filtered = f"static/processed/denoise/filter-{name}.png"
    cv2.imwrite(save_path_filtered, image_filtered)

    # gaussian blur
    gaussian_filtered = cv2.GaussianBlur(image,(5,5),sigmaX=4,sigmaY=4)
    save_path_gaussian = f"static/processed/denoise/gaussian-{name}.png"
    cv2.imwrite(save_path_gaussian, gaussian_filtered)

    # median blur
    median_filtered = cv2.medianBlur(image, 5)
    save_path_median = f"static/processed/denoise/median-{name}.png"
    cv2.imwrite(save_path_median, median_filtered)

    # mean blur
    mean_filtered = cv2.blur(image, (5, 5))
    save_path_mean = f"static/processed/denoise/mean-{name}.png"
    cv2.imwrite(save_path_mean, mean_filtered)

def sharpen(image, name):
    strong_kernel = np.array([[-1, -1, -1],
                              [-1, 10, -1],
                              [-1, -1, -1]])
    kernel = np.array([[-1,-1,-1], 
                    [-1, 9,-1],
                    [-1,-1,-1]])

    light_kernel = np.array([[ 0, -1,  0],
                                [-1,  5, -1],
                                [ 0, -1,  0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    save_path = f"static/processed/sharpen/standard-{name}.png"
    cv2.imwrite(save_path, sharpened)

    strong_sharpened = cv2.filter2D(image, -1, strong_kernel)
    save_path_strong = f"static/processed/sharpen/strong-{name}.png"
    cv2.imwrite(save_path_strong, strong_sharpened)

    light_sharpened = cv2.filter2D(image, -1, light_kernel)
    save_path_light= f"static/processed/sharpen/light-{name}.png"
    cv2.imwrite(save_path_light, light_sharpened)

def edge_detect(image, name):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # sobel
    ddepth = cv2.CV_64F
    sobel_x = cv2.Sobel(src=img_gray, ddepth=ddepth, dx=1, dy=0, ksize=3)
    save_sobel_x= f"static/processed/edge-detect/sobelx-{name}.png"
    cv2.imwrite(save_sobel_x, sobel_x)

    sobel_y = cv2.Sobel(src=img_gray, ddepth=ddepth, dx=0, dy=1, ksize=3)
    save_sobel_y= f"static/processed/edge-detect/sobely-{name}.png"
    cv2.imwrite(save_sobel_y, sobel_y)

    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel = cv2.convertScaleAbs(sobel_combined)
    save_sobel= f"static/processed/edge-detect/sobel-{name}.png"
    cv2.imwrite(save_sobel, sobel)

    # Prewitt
    prewitt_x_kernel = np.array([[1, 0, -1],
                                [1, 0, -1],
                                [1, 0, -1]])
    prewitt_x = cv2.filter2D(img_gray, -1, prewitt_x_kernel)
    save_prewitt_x= f"static/processed/edge-detect/prewittx-{name}.png"
    cv2.imwrite(save_prewitt_x, prewitt_x)

    prewitt_y_kernel = np.array([[1, 1, 1],
                                [0, 0, 0],
                                [-1, -1, -1]])

    prewitt_y = cv2.filter2D(img_gray, -1, prewitt_y_kernel)
    save_prewitt_y= f"static/processed/edge-detect/prewitty-{name}.png"
    cv2.imwrite(save_prewitt_y, prewitt_y)

    prewitt = cv2.magnitude(prewitt_x.astype(float), prewitt_y.astype(float))
    save_prewitt= f"static/processed/edge-detect/prewitt-{name}.png"
    cv2.imwrite(save_prewitt, prewitt)

    # Canny
    canny = cv2.Canny(img_gray, 100, 200)
    save_canny= f"static/processed/edge-detect/canny-{name}.png"
    cv2.imwrite(save_canny, canny)