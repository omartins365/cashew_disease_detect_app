import os
import subprocess

YOLOV10_DIR = "app/yolov10"
YOLOV10_REPO = "https://github.com/THU-MIG/yolov10.git"

def main():
    if not os.path.exists(YOLOV10_DIR):
        print(f"Cloning YOLOv10 into {YOLOV10_DIR}...")
        subprocess.check_call(["git", "clone", YOLOV10_REPO, YOLOV10_DIR])
    else:
        print(f"{YOLOV10_DIR} already exists. Skipping clone.")

    print("Installing YOLOv10 as editable package...")
    subprocess.check_call(["pip", "install", "-e", YOLOV10_DIR])

if __name__ == "__main__":
    main()