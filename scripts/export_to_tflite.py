# export_to_tflite.py
"""Convert YOLOv8n to an INT8-quantized TFLite model.

Calibration happens during export. The data YAML points to representative CCTV
frames extracted from the deployment video, which is better than generic COCO
calibration for this project.
"""

from ultralytics import YOLO


PT_PATH = "models/yolov8n.pt"
DATA_YAML = "configs/calib_dataset.yaml"


model = YOLO(PT_PATH)
model.export(
    format="tflite",
    imgsz=640,
    int8=True,
    data=DATA_YAML,
)

print("TFLite INT8 export completed. Check the export folder for a .tflite file.")
