# Object Detection Yolov10 Streamlit

This project integrates the YOLOv10 model for object detection into a Streamlit application. The app allows users to upload images or videos and receive object detection results. It provides an interactive interface for users to interact with the model and visualize the results directly in their browser.

### Files and Directories

- **app**: Main application code.
  - **app.py**: Streamlit app for uploading and processing images/videos.
  - **config.py**: Configuration settings, including paths and URLs.
  - **utils.py**: Utility functions, such as model downloading and image processing.
  - **train.py**: Script to train a custom YOLOv10 model.
- **.devcontainer**: Development container configuration.
  - **devcontainer.json**: Setup configuration for the development container.
  - **Dockerfile**: Dockerfile for the development container image.
- **requirements.txt**: Python dependencies for the project.

## Installation

Follow these steps to run the project locally:

### 1. Clone the Repository

```bash
https://github.com/omartins365/cashew_disease_detect_app.git
cd object-detection-yolov10-streamlit
```

### 2. Set Up the Development Container

This project uses a development container for consistent environments. Ensure Docker and Visual Studio Code with the Remote - Containers extension are installed.

1. Open the project in Visual Studio Code.
2. Press F1, type `Remote-Containers: Open Folder in Container...`, and select the project folder.

### 3. Install Dependencies

Without the development container, install dependencies manually:

```bash
pip install -r requirements.txt
git clone https://github.com/THU-MIG/yolov10.git app/yolov10
pip install -r app/yolov10/requirements.txt
pip install -e app/yolov10
```

### 4. Train Custom Model

Adjust `config.py` with your settings (`BATCH_SIZE`, `EPOCHS`, `IMG_SIZE`, `YAML_PATH`). Then run:

```bash
python app/train.py
```

### 5. Run the Application

```bash
streamlit run app/app.py
```

Access the application at http://localhost:8501

## Usage


1. **Select Type**: Choose `Image` or `Video` for your upload.
2. **Upload File**: Use the sidebar file uploader to upload an image (JPG, PNG, JPEG) or video (MP4, AVI, MOV).
3. **View and Process**:
   - For **Image**: Display and process the uploaded image.
   - For **Video**: Display and process the uploaded video (WEBM format).
4. **Result**: Processed files are saved in `app/result`. View them in the app interface.

## Notes

- Ensure dependencies are installed as listed in requirements.txt.
- For custom models, ensure `config.py` is correctly configured.
- Demo files for testing are available [here](https://drive.google.com/drive/folders/19fTHmV19c-q4PiMj1Gd4SPuN19HYPHi9?usp=sharing).
