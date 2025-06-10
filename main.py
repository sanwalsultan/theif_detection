from PIL import Image
import streamlit as st
import os
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
MODEL_PATH = "epoch-30.pt"  # Replace with your model path
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Failed to load the YOLO model: {e}")
    st.stop()

# Streamlit app setup
st.set_page_config(page_title="Thief Detection System", page_icon="🚨", layout="centered")
st.title("🚨 Thief Detection System")
st.markdown("""
This system detects suspicious activities such as hiding faces, carrying weapons, or breaking doors.  
Upload an **image** or **video**, and the model will analyze it.  
*Supported formats: JPG, JPEG, PNG, MP4*
""")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your file:", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    file_extension = os.path.splitext(uploaded_file.name)[-1].lower()

    # Handle image files
    if file_extension in [".jpg", ".jpeg", ".png"]:
        st.markdown("### 📷 Uploaded Image")
        try:
            # Display the image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Convert image to array and predict
            image_array = np.array(image)
            st.markdown("### 🔍 Running Detection...")
            results = model.predict(source=image_array, conf=0.25, save=False)

            # Display the results
            st.image(results[0].plot(), caption="Detection Results", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error analyzing the image: {e}")

    # Handle video files
    elif file_extension == ".mp4":
        st.markdown("### 🎥 Uploaded Video")
        try:
            # Save the video locally
            temp_video_path = "uploaded_video.mp4"
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.read())
            st.video(temp_video_path)

            # Run detection
            st.markdown("### 🔍 Running Detection on Video...")
            results = model.predict(source=temp_video_path, conf=0.25, save=True)

            # Notify the user
            st.success("✅ Detection complete! Check the saved output in the model directory.")
        except Exception as e:
            st.error(f"❌ Error analyzing the video: {e}")
        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

    else:
        st.error("❌ Unsupported file format. Please upload a valid image or video.")
else:
    st.info("📤 Please upload a file to begin the detection process.")
