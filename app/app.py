import os
import subprocess

if not os.path.exists("app/yolov10"):
    subprocess.run(["git", "clone", "https://github.com/THU-MIG/yolov10.git", "app/yolov10"], check=True)

import logging  # Add this import
import os
import tempfile
import platform
import psutil

import av
import cv2
import numpy as np
import streamlit as st
from config import (
    DEMO_FILES_URL,
    IMAGE_WIDTH,
    MODEL_PATH,
    RESULT_PATH,
    TRAINED_MODEL_PATH,
)
from PIL import Image
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLOv10
from utils import download_model, load_model
import torch
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def process_image(model: YOLOv10, image_path: str, result_path: str) -> dict:
    """ Perform object detection on an image, save the result, and return performance metrics. """
    try:
        logging.info(f"Processing image: {image_path}")
        start_time = time.time()
        result = model(source=image_path)  # Perform object detection
        inference_time = time.time() - start_time
        result[0].save(result_path)  # Save the result image
        logging.info(f"Result saved: {result_path} in {inference_time:.2f} seconds")
        return {"success": True, "inference_time": inference_time, "result": result}
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        st.error(f"Error processing image: {e}")
        return {"success": False, "inference_time": None}


def process_video(model: YOLOv10, video_path: str, result_path: str) -> dict:
    """Perform object detection on a video, save the result, and return performance metrics."""
    start_time = time.time()
    try:
        logging.info(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error("Could not open video file.")
            st.error("Error: Could not open video file.")
            return {"success": False, "inference_time": None}

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # fallback fps

        fourcc = cv2.VideoWriter_fourcc(*'vp80')
        out = cv2.VideoWriter(result_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        interval = int(fps * 30)  # run detection every 30 seconds
        last_results = None

        logging.info(f"Video properties: width={frame_width}, height={frame_height}, fps={fps}, interval={interval}")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                logging.info("End of video or failed to read frame.")
                break

            if frame_count % interval == 0 or last_results is None:
                logging.info(f"Running detection at frame {frame_count}")
                last_results = model(frame)

            out.write(last_results[0].plot())
            frame_count += 1

        inference_time = time.time() - start_time
        logging.info(f"Video processing complete. Output saved: {result_path} in {inference_time:.2f} seconds")
        return {"success": True, "inference_time": inference_time}
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        st.error(f"Error processing video: {e}")
        return {"success": False, "inference_time": None}
    finally:
        cap.release()
        out.release()


def select_model() -> YOLOv10:
    """Select and load the model based on user choice."""
    model_choice = "Custom Trained Model";
    # st.sidebar.selectbox("Select model", ["Default YOLOv10", "Custom Trained Model"])
    model_path = TRAINED_MODEL_PATH if model_choice == "Custom Trained Model" else MODEL_PATH
    
    if model_choice == "Default YOLOv10" and not download_model(MODEL_PATH):
        return None

    if model_choice == "Custom Trained Model" and not os.path.exists(TRAINED_MODEL_PATH):
        st.error(f"Model file '{TRAINED_MODEL_PATH}' not found.")
        return None

    return load_model(model_path)

def has_gpu():
    return torch.cuda.is_available()

def display_and_process_file(model: YOLOv10, type_choice: str, temp_path: str, result_path: str) -> None:
    """ Process the uploaded file based on the selected type (Image or Video). """
    try:
        if type_choice == "Image":
            image = Image.open(temp_path)
            image_size = os.path.getsize(temp_path) / 1024  # KB
            image_width, image_height = image.size

            # Device info
            device_info = {
                "Platform": platform.platform(),
                # "Processor": platform.processor(),
                # "RAM (GB)": round(psutil.virtual_memory().total / (1024 ** 3), 2),
                "Inference via": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            }
            st.image(image, "Uploaded Image", width=IMAGE_WIDTH)
            # Process and display result image
            with st.spinner("Processing image..."):
                result = process_image(model, temp_path, result_path)
                if result["success"]:
                    st.success(f"Image processed. Result saved: {result_path}")
                    st.image(result_path, "Result Image", width=IMAGE_WIDTH)
                    # st.write(f"**Inference Time:** {result['inference_time']:.2f} seconds")
                    st.metric("Inference Time (s)", f"{result['inference_time']:.2f}")
                    st.write(f"**Image Size:** {image_width}x{image_height} px, {image_size:.1f} KB")
                    st.write("**Device Info:**")
                    st.json(device_info)
                else:
                    st.error("Image processing failed.")
        else:
            st.video(temp_path)
            cap = cv2.VideoCapture(temp_path)
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_size = os.path.getsize(temp_path) / 1024  # KB
            cap.release()

     
            # Device info
            device_info = {
                "Platform": platform.platform(),
                # "Processor": platform.processor(),
                # "RAM (GB)": round(psutil.virtual_memory().total / (1024 ** 3), 2),
                "Inference via": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            }

            # Process and display result video
            with st.spinner("Processing video..."):
                result_path = result_path.replace(".mp4", ".webm")
                result = process_video(model, temp_path, result_path)
                if result["success"]:
                    st.success(f"Video processed. Result saved: {result_path}")
                    st.video(result_path)
                    st.write(f"**Total Processing Time:** {result['total_time']:.2f} seconds")
                    st.metric("Total Processing Time (s)", f"{result['total_time']:.2f}")
                    st.metric("Average FPS", f"{result['fps']:.2f}")
                    st.write(f"**Video Size:** {video_width}x{video_height} px, {video_fps} FPS, {video_frames} frames, {video_size:.1f} KB")
                    st.write("**Device Info:**")
                    st.json(device_info)
                else:
                    st.error("Video processing failed.")
    except Exception as e:
        st.error(f"Error during processing: {e}")


def display_and_process_camera(model: YOLOv10) -> None:
    """Capture and process an image from the user's webcam."""
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=RESULT_PATH) as temp_file:
            temp_file.write(camera_image.getvalue())
            temp_path = temp_file.name

        result_path = os.path.join(RESULT_PATH, "result_camera.jpg")
        with st.spinner("Processing camera image..."):
            if process_image(model, temp_path, result_path):
                st.success(f"Camera image processed. Result saved: {result_path}")
                st.image(result_path, "Result Image", width=IMAGE_WIDTH)
        os.remove(temp_path)


def draw_boxes(frame: np.ndarray, results) -> np.ndarray:
    """Draw bounding boxes from YOLO results on the frame."""
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{results[0].names[cls]} {conf:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


def live_cam_detect(model: YOLOv10):
    st.title("Live Cam Detect")
    st.info("Allow camera access and see real-time detection.")

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        img = results[0].plot()  # Use YOLO's built-in plotting
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="live-detect",
        video_frame_callback=callback,
        sendback_audio=False
    )


def main():
    
    st.sidebar.title("Cashew Deasease Detection")

    # st.sidebar.title("🌰 Cashew Disease Detection")
    st.sidebar.markdown("Upload an image or video of cashew leaves, nuts or stem to detect diseases using the custom YOLOv10 model.")

    model = select_model()
    
    if model is None:
        st.markdown("""
        # 🌰 Cashew Disease Detection (YOLOv10)
        Detect diseases in cashew plants using a custom-trained YOLOv10 model.  
        Upload an image or video, or use your camera to get instant results.
        """)
        return

    type_options = ["Image", "Video", "Camera"]
    if has_gpu():
        type_options.append("Live Cam Detect")
    type_choice = st.sidebar.selectbox(
        "Select Input Type", type_options
    )
    file = None

    if type_choice == "Image":
        st.sidebar.markdown("🖼️ **Upload a clear image of a cashew leaf or nut.**")
    elif type_choice == "Video":
        st.sidebar.markdown("🎥 **Upload a short video showing cashew plants.**")
    elif type_choice == "Camera":
        st.sidebar.markdown("📷 **Capture a photo using your device camera.**")
    elif type_choice == "Live Cam Detect":
        st.sidebar.markdown("🔴 **Live detection (GPU required).**")

    if type_choice in ["Image", "Video"]:
        file = st.sidebar.file_uploader(
            "Choose a file...",
            type=["jpg", "png", "jpeg"] if type_choice == "Image" else ["mp4", "avi", "mov"]
        )

    if type_choice == "Camera":
        os.makedirs(RESULT_PATH, exist_ok=True)
        display_and_process_camera(model)
    elif type_choice == "Live Cam Detect":
        live_cam_detect(model)
    elif file:
        os.makedirs(RESULT_PATH, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}", dir=RESULT_PATH) as temp_file:
            temp_file.write(file.getvalue())
            temp_path = temp_file.name

        result_path = os.path.join(RESULT_PATH, f"result_{file.name}")
        display_and_process_file(model, type_choice, temp_path, result_path)
        os.remove(temp_path)
    else:
        if type_choice in ["Image", "Video"]:
            st.sidebar.markdown(f"You can download demo files [here]({DEMO_FILES_URL}).")


if __name__ == "__main__":
    main()
