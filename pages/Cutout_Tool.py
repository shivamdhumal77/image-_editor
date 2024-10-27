import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import io

# Load MediaPipe's selfie segmentation model
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Function to create a smooth foreground cutout with blurred edges and a transparent background
def create_smooth_foreground_cutout(image, mask_threshold=0.3, edge_blur_intensity=15):
    # Convert image to RGB (as required by MediaPipe) only if necessary
    if image.shape[2] == 4:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use MediaPipe selfie segmentation model to get mask
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as model:
        results = model.process(image_rgb)
        mask = results.segmentation_mask

    # Create a binary mask based on the threshold
    foreground_mask = (mask > mask_threshold).astype(np.uint8) * 255

    # Smooth the edges with Gaussian blur
    edge_blur_intensity = max(1, edge_blur_intensity)
    kernel_size = edge_blur_intensity if edge_blur_intensity % 2 == 1 else edge_blur_intensity + 1
    blurred_edges = cv2.GaussianBlur(foreground_mask, (kernel_size, kernel_size), 0)

    # Convert mask to 4 channels (RGBA) for transparency
    cutout_with_alpha = cv2.merge((image[:, :, 0], image[:, :, 1], image[:, :, 2], blurred_edges))

    return cutout_with_alpha

# Streamlit UI setup
st.title("AI Cutout Tool")
st.write("Upload an image to remove the background, keeping only the subject with smooth, blurred edges.")

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Load and display the original image
    image = np.array(Image.open(uploaded_image).convert("RGBA"))
    st.image(image, caption="Original Image Preview", use_column_width=True)

    # Edge blur intensity slider
    edge_blur_intensity = st.slider("Edge Blur Intensity", 1, 50, 15)

    # Button to create cutout with blurred edges
    if st.button("Create Cutout with Blurred Edges"):
        # Generate cutout with smooth edges and transparency
        cutout_image_with_blur = create_smooth_foreground_cutout(image, mask_threshold=0.3, edge_blur_intensity=edge_blur_intensity)

        # Display the cutout preview
        st.image(cutout_image_with_blur, caption="Smooth Cutout with Blurred Edges", use_column_width=True)

        # Save the cutout as a PNG
        cutout_pil = Image.fromarray(cutout_image_with_blur)
        cutout_buffer = io.BytesIO()
        cutout_pil.save(cutout_buffer, format="PNG")

        # Download button for the cutout image
        st.download_button(
            label="Download Cutout with Blurred Edges as PNG",
            data=cutout_buffer.getvalue(),
            file_name="cutout_with_blurred_edges.png",
            mime="image/png"
        )
