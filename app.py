import streamlit as st
import cv2
import numpy as np
import face_recognition
from PIL import Image, ImageDraw
import tempfile
import os

st.set_page_config(page_title="OpenCV Face Detection", layout="wide")

st.title("OpenCV Face Detection & Landmarks")
st.markdown("Upload an image to detect faces and landmarks using `face_recognition` (HOG/CNN).")

# Sidebar Configuration
st.sidebar.header("Settings")

# Detection Model
model_type = st.sidebar.radio(
    "Detection Model",
    ("hog", "cnn"),
    index=0,
    help="HOG is faster (CPU), CNN is more accurate (GPU recommended but works on CPU)."
)

# Upsampling
upsample_times = st.sidebar.slider(
    "Upsample Times",
    min_value=0,
    max_value=2,
    value=1,
    help="Higher values find smaller faces but take longer."
)

# Visualization Options
st.sidebar.subheader("Visualization")
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)
box_color = st.sidebar.color_picker("Box Color", "#00FF00")
box_thickness = st.sidebar.slider("Box Thickness", 1, 10, 3)

show_landmarks = st.sidebar.checkbox("Show Landmarks", value=False)
landmark_color = st.sidebar.color_picker("Landmark Color", "#FF0000")
landmark_width = st.sidebar.slider("Landmark Width", 1, 5, 1)

def hex_to_rgb(hex_color):
    """Converts hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def process_image(uploaded_file):
    # Load image from file-like object
    image = face_recognition.load_image_file(uploaded_file)
    
    # Detect faces
    with st.spinner(f"Detecting faces using {model_type.upper()} model..."):
        face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=upsample_times, model=model_type)
    
    # Get landmarks if requested
    face_landmarks_list = []
    if show_landmarks:
        with st.spinner("Extracting landmarks..."):
            face_landmarks_list = face_recognition.face_landmarks(image, face_locations)

    # Convert to PIL for drawing
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    rgb_box_color = hex_to_rgb(box_color)
    rgb_landmark_color = hex_to_rgb(landmark_color)

    # Draw boxes
    if show_boxes:
        for top, right, bottom, left in face_locations:
            draw.rectangle(((left, top), (right, bottom)), outline=rgb_box_color, width=box_thickness)

    # Draw landmarks
    if show_landmarks:
        for landmarks in face_landmarks_list:
            for facial_feature in landmarks.keys():
                draw.line(landmarks[facial_feature], fill=rgb_landmark_color, width=landmark_width)

    return pil_image, len(face_locations)

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(uploaded_file, use_container_width=True)

    # Process
    if st.button("Detect Faces") or (show_boxes or show_landmarks):
        try:
            # Reset file pointer to beginning before reading again
            uploaded_file.seek(0)
            processed_image, face_count = process_image(uploaded_file)
            
            with col2:
                st.subheader("Result")
                st.image(processed_image, caption=f"Detected {face_count} face(s)", use_container_width=True)
                
            if face_count > 0:
                st.success(f"Found {face_count} face(s)!")
            else:
                st.warning("No faces found. Try increasing 'Upsample Times' or using CNN model.")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
