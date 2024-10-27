import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import io

# Load MediaPipe's selfie segmentation model
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Function to apply black-and-white effect to background and keep the foreground colorful
def color_background_bw(image, brightness=1.0, contrast=1.0, saturation=1.0):
    # Convert image to RGB for MediaPipe processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform background segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as model:
        results = model.process(image_rgb)
        mask = results.segmentation_mask > 0.1  # Foreground threshold

    # Convert image to grayscale for the background
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # Apply brightness and contrast adjustments to the grayscale background
    gray_background = cv2.convertScaleAbs(gray_background, alpha=contrast, beta=brightness * 50)

    # Convert grayscale background to HSV for saturation adjustments
    hsv_background = cv2.cvtColor(gray_background, cv2.COLOR_BGR2HSV)
    hsv_background[..., 1] = np.clip(hsv_background[..., 1] * saturation, 0, 255)  # Adjust saturation
    gray_background = cv2.cvtColor(hsv_background, cv2.COLOR_HSV2BGR)

    # Convert mask to 3 channels (same as the image)
    mask_3d = np.dstack((mask, mask, mask))

    # Apply the mask: keep the subject in color and set the background to black and white
    output_image = np.where(mask_3d, image, gray_background)

    return output_image

# Streamlit UI
st.title("Selective Black-and-White Background Tool with Saturation Control")
st.write("Upload an image to apply a black-and-white effect to the background while keeping the foreground colorful.")

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Load image using PIL (ensures correct color for preview)
    image = np.array(Image.open(uploaded_image))
    
    # Display the original image (in correct color for Streamlit)
    st.image(image, caption="Original Image Preview", use_column_width=True)

    # Convert the image to BGR for OpenCV processing (only needed for OpenCV functions)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Sliders for brightness, contrast, and saturation adjustments for background
    brightness = st.slider("Background Brightness", 0.5, 2.0, 1.0)
    contrast = st.slider("Background Contrast", 0.5, 2.0, 1.0)
    saturation = st.slider("Background Saturation", 0.0, 2.0, 1.0)

    # Button to apply black-and-white effect to background
    if st.button("Apply Background Black-and-White Effect"):
        # Apply color background black-and-white effect with saturation control
        output_image_bgr = color_background_bw(image_bgr, brightness, contrast, saturation)

        # Convert the output back to RGB for displaying in Streamlit
        output_image_rgb = cv2.cvtColor(output_image_bgr, cv2.COLOR_BGR2RGB)

        # Display the result
        st.image(output_image_rgb, caption="Image with Black-and-White Background", use_column_width=True)

        # Convert the result to PNG for download
        buffered = io.BytesIO()
        result = Image.fromarray(output_image_rgb)
        result.save(buffered, format="PNG")

        # Download button for the processed image
        st.download_button(
            label="Download Image with Black-and-White Background",
            data=buffered.getvalue(),
            file_name="bw_background_output.png",
            mime="image/png"
        )
