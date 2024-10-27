import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import io

# Load MediaPipe's selfie segmentation model
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Function to blur the background but keep the subject clear
def blur_background(image, blur_intensity):
    # Convert image to RGB for MediaPipe processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform background segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as model:
        results = model.process(image_rgb)
        mask = results.segmentation_mask > 0.1  # Foreground threshold

    # Create a blurred version of the image
    blur_kernel_size = blur_intensity * 4 + 1  # Increased kernel size for stronger blur effect
    blurred_image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)

    # Convert mask to 3 channels (same as the image)
    mask_3d = np.dstack((mask, mask, mask))

    # Apply the mask: keep the subject and blur the background
    output_image = np.where(mask_3d, image, blurred_image)

    return output_image

# Streamlit UI
st.title("AI Background Blurring Tool with Clean Edges")
st.write("Upload an image and blur the background while keeping the subject clear.")

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Load image using PIL (ensures correct color for preview)
    image = np.array(Image.open(uploaded_image))
    
    # Display the original image (in correct color for Streamlit)
    st.image(image, caption="Original Image Preview", use_column_width=True)

    # Convert the image to BGR for OpenCV processing (only needed for OpenCV functions)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Slider for blur intensity (1 to 20)
    blur_intensity = st.slider("Blur Intensity", 1, 20, 10)

    # Recommendation for realistic blur
    st.write("Recommendation: For a DSLR-like blur effect, keep the blur intensity around 5-10; for more pronounced effects, use higher values.")

    # Button to apply background blur
    if st.button("Apply Background Blur"):
        # Apply background blur with clean edges
        output_image_bgr = blur_background(image_bgr, blur_intensity)

        # Convert the output back to RGB for displaying in Streamlit
        output_image_rgb = cv2.cvtColor(output_image_bgr, cv2.COLOR_BGR2RGB)

        # Display the result
        st.image(output_image_rgb, caption="Image with Blurred Background", use_column_width=True)

        # Convert the result to PNG for download
        buffered = io.BytesIO()
        result = Image.fromarray(output_image_rgb)
        result.save(buffered, format="PNG")

        # Download button for the blurred image
        st.download_button(
            label="Download Blurred Image in HD",
            data=buffered.getvalue(),
            file_name="blurred_output.png",
            mime="image/png"
        )
