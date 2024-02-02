import gradio as gr
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# draw a image's upper, left, and right border in black
def draw_border(img):
    height, width = img.shape

    line_thickness = 1
    line_color = (0, 0, 0)

    cv.line(img, (0, 0), (width, 0), line_color, line_thickness)
    cv.line(img, (0, 0), (0, height), line_color, line_thickness)
    cv.line(img, (width-1, 0), (width-1, height), line_color, line_thickness)

def mark_sky(original_image):
    # Convert to grayscale
    img_gray = cv.cvtColor(original_image, cv.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    img_blurred = cv.GaussianBlur(img_gray, (21, 5), 0)

    # Use Adaptive Gaussian Thresholding to get a binary mask
    img_binary = cv.adaptiveThreshold(img_blurred,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

    # apply morphological operation opening to remove noise
    # erosion and dilation, twice
    kernel = np.ones((6, 6), np.uint8)
    img_opened = cv.morphologyEx(img_binary, cv.MORPH_OPEN, kernel, iterations=2)

    # draw the image's upper, left, and right border in black
    # to conveniently draw sky contour later
    draw_border(img_opened)

    # find contours in opened and border-drawn image
    contours, _ = cv.findContours(img_opened, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # find the largest contour among those close to upper border
    # 'close' means the countour has intersection with the upper 10% of the image
    upper_contours = [cnt for cnt in contours if cv.boundingRect(cnt)[1] < original_image.shape[0] * 0.1]
    largest_upper_contour = max(upper_contours, key=cv.contourArea)

    # draw the largest contour in red on the original image
    cv.drawContours(original_image, [largest_upper_contour], -1, (255, 51, 51), thickness=cv.FILLED)


    return original_image

# Gradio interface
demo = gr.Interface(
    fn=mark_sky,
    inputs = gr.Image(),
    outputs = gr.Image(),
    description = "Upload an image to get the sky marked in red.",
)

demo.launch()