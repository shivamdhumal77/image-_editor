import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def adjust_image(image, brightness=0, contrast=0, saturation=0, smoothness=0, warmth=0):
    # Convert to float for adjustment
    img = image.astype(np.float32)

    # Adjust brightness
    img += brightness

    # Adjust contrast using the centered formula
    if contrast != 0:
        contrast_factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
        img = contrast_factor * (img - 128) + 128

    # Clip values to stay in the valid range
    img = np.clip(img, 0, 255)

    # Convert back to uint8
    img = img.astype(np.uint8)

    # Convert to HSV for saturation adjustment
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] * (1 + saturation / 100), 0, 255)

    # Convert back to BGR
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    # Apply smoothness effect only to the face
    if smoothness > 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_region = img[y:y + h, x:x + w]
            smoothed_face = cv2.bilateralFilter(face_region, d=9, sigmaColor=smoothness * 10, sigmaSpace=smoothness * 10)
            img[y:y + h, x:x + w] = smoothed_face

    # Adjust warmth and coldness
    if warmth > 0:
        img[..., 0] = np.clip(img[..., 0] + warmth * 0.5, 0, 255)
        img[..., 1] = np.clip(img[..., 1] + warmth * 0.3, 0, 255)
    elif warmth < 0:
        img[..., 2] = np.clip(img[..., 2] - abs(warmth) * 0.5, 0, 255)

    return img

# Streamlit interface
st.title("Photo Enhancement Tool")
st.write("Upload an image and adjust the sliders to enhance it.")

uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image = np.array(image)

    st.image(image, caption='Original Image', use_column_width=True)

    brightness = st.slider("Brightness", -100, 100, 0)
    contrast = st.slider("Contrast", -100, 100, 0)
    saturation = st.slider("Saturation", -100, 100, 0)
    smoothness = st.slider("Smoothness (Face Only)", 0, 10, 0)
    warmth = st.slider("Warmth / Coldness", -100, 100, 0)

    enhanced_image = adjust_image(image, brightness, contrast, saturation, smoothness, warmth)
    st.image(enhanced_image, caption='Enhanced Image', width=500)

    # Convert enhanced image to PIL format for download
    enhanced_pil = Image.fromarray(enhanced_image)
    buffered = io.BytesIO()
    enhanced_pil.save(buffered, format="PNG")
    buffered.seek(0)

    # Download link
    st.download_button(
        label="Download Enhanced Image",
        data=buffered,
        file_name="enhanced_image.png",
        mime="image/png"
    )
