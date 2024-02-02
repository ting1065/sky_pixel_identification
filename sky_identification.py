import gradio as gr
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def mark_sky(img):
    # Convert to grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    img_blurred = cv.GaussianBlur(img_gray, (21, 5), 0)

    # Use Adaptive Gaussian Thresholding to get a binary mask
    img_binary = cv.adaptiveThreshold(img_blurred,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

    # apply morphological operation opening to remove noise
    # erosion and dilation, twice
    kernel = np.ones((6, 6), np.uint8)
    img_opened = cv.morphologyEx(img_binary, cv.MORPH_OPEN, kernel, iterations=2)

    return img_opened

# Gradio interface
demo = gr.Interface(
    fn=mark_sky,
    inputs = gr.Image(),
    outputs = gr.Image(label="Processed Image"),
    description = "Upload an image to get the sky marked in red.",
)

demo.launch()