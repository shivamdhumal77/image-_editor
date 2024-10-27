import streamlit as st
import cv2
import numpy as np
from PIL import Image

def apply_color_adjustments(image, color, exposure=0, saturation=1.0, contrast=1.0):
    """
    Applies selective color adjustments to an image.

    Args:
        image: The input image as a NumPy array.
        color: The color to adjust (e.g., "red", "green").
        exposure: Exposure adjustment value (positive for brighter, negative for darker).
        saturation: Saturation adjustment value (0 for grayscale, >1 for more saturation).
        contrast: Contrast adjustment value (0.5 for lower, 2.0 for higher).

    Returns:
        The adjusted image as a NumPy array.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    color_ranges = {
        "red": [(0, 100, 100), (10, 255, 255), (160, 100, 100), (180, 255, 255)],
        "green": [(35, 100, 100), (85, 255, 255)],
        "yellow": [(20, 100, 100), (30, 255, 255)],
        "blue": [(90, 100, 100), (130, 255, 255)],
        "violet": [(130, 100, 100), (160, 255, 255)],
        "orange": [(10, 100, 100), (20, 255, 255)]
    }

    if color not in color_ranges:
        st.error("Selected color is not available. Choose a different color.")
        return image

    mask = np.zeros_like(hsv[:, :, 0])
    for lower, upper in color_ranges[color]:
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

    hsv[..., 2] = np.where(mask, np.clip(hsv[..., 2] + exposure, 0, 255), hsv[..., 2])
    hsv[..., 1] = np.where(mask, np.clip(hsv[..., 1] * saturation, 0, 255), hsv[..., 1])

    adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # Change to BGR for download
    return adjusted_image

# Streamlit App Layout
st.title("Selective Color Adjustment Photo Editor")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image = np.array(Image.open(uploaded_image))
    st.image(image, caption="Original Image", use_column_width=True)

    color = st.selectbox("Select a color to adjust", ("red", "green", "yellow", "blue", "violet", "orange"))
    exposure = st.slider("Exposure", -100, 100, 0)
    saturation = st.slider("Saturation", 0.0, 3.0, 1.0)
    contrast = st.slider("Contrast", 0.5, 2.0, 1.0)

    adjusted_image = apply_color_adjustments(image, color, exposure, saturation, contrast)
    st.image(adjusted_image, caption="Adjusted Image with Selective Color Adjustment", use_column_width=True)

    # Convert adjusted image to bytes for download
    _, buffer = cv2.imencode('.png', adjusted_image)
    adjusted_image_bytes = buffer.tobytes()

    # Download button
    st.download_button(
        label="Download Adjusted Image",
        data=adjusted_image_bytes,
        file_name="adjusted_image.png",
        mime="image/png"
    )
