## Lab REQUIREMENT: Image matching with Histogram
1. Students download the image dataset provided by the instructor on the course website.
The dataset consists of two sets: one test set and one query set.
2. Write a Python app that takes an input image from the test set, then finds the 10 most similar images to the input image using Histogram for image matching.

## HOW TO RUN
1. Install all dependencies:
````bash
pip install -r requirements.txt
````
2. Download Dataset and place in "static/dataset/": https://drive.google.com/file/d/1F6sPtl0H-Sh7XPrAojDKcz_rBoUl_fgu/view?usp=sharing

3. Run server:
````bash
uvicorn app:app
````

## HOW TO USE
1. Upload an query image:
![Query Image](./demo/upload-image.png)

2. Choose an calculate methods and Find 10 images:
![Find 10 images](./demo/retrieve.png)