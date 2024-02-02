import gradio as gr
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def mark_sky(img):
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    return img

# Gradio interface
demo = gr.Interface(
    fn=mark_sky,
    inputs = gr.Image(),
    outputs = gr.Image(label="Processed Image"),
    description = "Upload an image to get the sky marked in red.",
)

demo.launch()