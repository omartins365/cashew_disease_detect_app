# Cashew Disease Detection and Classification System (YOLOv10)

## Project Structure

```
cashew_disease_detect_app/
├── app/
│   ├── app.py
│   ├── config.py
│   ├── live.py
│   ├── train.py
│   ├── utils.py
│   ├── asset/
│   │   └── lautech_logo.png
│   ├── model/
│   │   ├── best.onnx
│   │   ├── best.pt
│   │   └── yolov10s.pt
│   └── yolov10/
│       ├── app.py
│       ├── ... (YOLOv10 source, docs, examples, etc.)
├── requirements.txt
├── README.md
├── setup.py
└── ... (other files and folders)
```

## Overview
This repository contains the implementation of a real-time Cashew Disease Detection and Classification System using the YOLOv10 deep learning model. The project is part of an M.Tech research thesis titled:

**"Development of a Real-Time Cashew Disease Detection And Classification System Using YOLO-V10"**

This application fulfills one of the core objectives of the research, providing an accessible, web-based tool for automated detection and classification of diseases in cashew plants (leaves, nuts, and stems) from images and videos.

---

## Live Demo
The application is deployed and accessible at:

👉 [https://cashew-disease-detection.streamlit.app](https://cashew-disease-detection.streamlit.app)

## Features
- **Image and Video Analysis:** Upload images or videos of cashew plants to detect and classify diseases in real time.
- **Camera Input:** Capture photos directly from your device for instant analysis.
- **Custom YOLOv10 Model:** Utilizes a custom-trained YOLOv10 model for high-accuracy detection and classification.
- **Performance Metrics:** Displays inference time, device information, and allows result downloads.
- **User-Friendly Interface:** Built with Streamlit for ease of use and accessibility.

## Research Context
This project is a practical deliverable for the M.Tech research work, aiming to:
- Develop a robust, real-time system for early detection of cashew diseases.
- Leverage state-of-the-art deep learning (YOLOv10) for agricultural disease management.
- Provide a tool that can assist farmers, researchers, and agricultural extension workers in disease monitoring and intervention.

## Getting Started
### Prerequisites
- Python 3.9+
- [Streamlit](https://streamlit.io/)
- [YOLOv10](https://github.com/THU-MIG/yolov10)
- See `requirements.txt` for all dependencies.

### Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/your-username/cashew_disease_detect_app.git
   cd cashew_disease_detect_app
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the application:
   ```sh
   streamlit run app/app.py
   ```

## Usage
- Upload an image or video of cashew leaves, nuts, or stems.
- Or use your device camera to capture a photo.
- The app will process the input and display detected disease regions and classification results.
- Download the processed results for further analysis or reporting.

## Acknowledgements
- YOLOv10: [https://github.com/THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)
- Streamlit: [https://streamlit.io/](https://streamlit.io/)
- Research supported by Ladoke Akintola University of Technology, Ogbomoso, Nigeria.
- Special thanks to my supervisors:
  - Engr. Prof. Mrs. A. O. Oke, B.Tech., M.Tech., Ph.D. (Professor of Computer Engineering, LAUTECH)
  - Engr. Prof. E. O. Omidiora, B.Tech., M.Tech., Ph.D. (Professor of Computer Engineering, LAUTECH)

## Citation
If you use this application or its results in your research, please cite appropriately.

---

For questions or collaboration, please contact: Martins Oladayo Ayoola (omartins365@gmail.com)
