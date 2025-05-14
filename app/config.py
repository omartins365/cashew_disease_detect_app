import os

MODEL_URL = "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt"
DEMO_FILES_URL = "https://drive.google.com/drive/folders/15mKocsFZ5L9EceynG5_x-Y6KvkE85pS0?usp=sharing"
MODEL_PATH = os.path.join(os.getcwd(), "app", "model", "yolov10n.pt")
RESULT_PATH = os.path.join(os.getcwd(), "app", "result")

IMAGE_WIDTH = 550

YAML_PATH = os.path.join(os.getcwd(), "app", "data", "data.yaml")

EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 24

TRAINED_MODEL_PATH = os.path.join(os.getcwd(), "app", "model", "best.onnx")