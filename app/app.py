import concurrent.futures
import logging  # Add this import
import os
import tempfile

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
from ultralytics import YOLOv10
from utils import download_model, load_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def process_image(model: YOLOv10, image_path: str, result_path: str) -> bool:
    """ Perform object detection on an image and save the result. """
    try:
        logging.info(f"Processing image: {image_path}")
        result = model(source=image_path)[0]  # Perform object detection
        result.save(result_path)  # Save the result image
        logging.info(f"Result saved: {result_path}")
        return True
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        st.error(f"Error processing image: {e}")
        return False


def process_video(model: YOLOv10, video_path: str, result_path: str) -> bool:
    """ Perform object detection on a video and save the result. """
    try:
        logging.info(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error("Could not open video file.")
            st.error("Error: Could not open video file.")
            return False

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'vp80')
        out = cv2.VideoWriter(result_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        interval = int(fps * 30)  # 10 seconds interval
        last_results = None
        future = None

        logging.info(f"Video properties: width={frame_width}, height={frame_height}, fps={fps}, interval={interval}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    logging.info("End of video or failed to read frame.")
                    break

                # Start async detection every interval or if first frame
                if frame_count % interval == 0 or last_results is None:
                    if future is not None:
                        # Wait for previous detection to finish
                        last_results = future.result()
                        logging.info(f"Detection finished at frame {frame_count}")
                    # Submit new detection task
                    logging.info(f"Submitting detection at frame {frame_count}")
                    future = executor.submit(model, frame)
                elif future is not None and future.done():
                    # If detection finished in background, update result
                    last_results = future.result()
                    future = None

                # Use last_results for plotting
                if last_results is not None:
                    out.write(last_results[0].plot())
                else:
                    out.write(frame)  # fallback: write raw frame

                frame_count += 1

            # Ensure last detection is written for remaining frames if needed
            if future is not None and not future.done():
                last_results = future.result()
                logging.info("Final detection finished at end of video.")

        logging.info(f"Video processing complete. Output saved: {result_path}")
        return True
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        st.error(f"Error processing video: {e}")
        return False
    finally:
        cap.release()
        out.release()


def select_model() -> YOLOv10:
    """Select and load the model based on user choice."""
    model_choice = st.sidebar.selectbox("Select model", ["Default YOLOv10", "Custom Trained Model"])
    model_path = TRAINED_MODEL_PATH if model_choice == "Custom Trained Model" else MODEL_PATH
    
    if model_choice == "Default YOLOv10" and not download_model(MODEL_PATH):
        return None

    if model_choice == "Custom Trained Model" and not os.path.exists(TRAINED_MODEL_PATH):
        st.error(f"Model file '{TRAINED_MODEL_PATH}' not found.")
        return None

    return load_model(model_path)


def display_and_process_file(model: YOLOv10, type_choice: str, temp_path: str, result_path: str) -> None:
    """ Process the uploaded file based on the selected type (Image or Video). """
    try:
        if type_choice == "Image":
            image = Image.open(temp_path)
            st.image(image, "Uploaded Image", width=IMAGE_WIDTH)
            # Process and display result image
            with st.spinner("Processing image..."):
                if process_image(model, temp_path, result_path):
                    st.success(f"Image processed. Result saved: {result_path}")
                    st.image(result_path, "Result Image", width=IMAGE_WIDTH)
        else:
            st.video(temp_path)
            # Process and display result video
            with st.spinner("Processing video..."):
                result_path = result_path.replace(".mp4", ".webm")
                if process_video(model, temp_path, result_path):
                    st.success(f"Video processed. Result saved: {result_path}")
                    st.video(result_path)
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


def main():
    st.sidebar.title("Cashew Deasease Detection")
    model = select_model()
    
    if model is None:
        return

    type_choice = st.sidebar.selectbox(
        "Select type", ["Image", "Video", "Camera"]
    )
    file = None
    if type_choice in ["Image", "Video"]:
        file = st.sidebar.file_uploader(
            "Choose a file...",
            type=["jpg", "png", "jpeg"] if type_choice == "Image" else ["mp4", "avi", "mov"]
        )

    if type_choice == "Camera":
        os.makedirs(RESULT_PATH, exist_ok=True)
        display_and_process_camera(model)
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
