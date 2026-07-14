# export_to_tflite.py
"""Convert YOLOv8n to an INT8-quantized TFLite model.

Calibration happens during export. The data YAML points to representative CCTV
frames extracted from the deployment video, which is better than generic COCO
calibration for this project.
"""

from ultralytics import YOLO
import argparse
import os


PT_PATH = "models/yolov8n.pt"
DATA_YAML = "configs/calib_dataset.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert YOLOv8n to an INT8-quantized TFLite model."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("VCTS_YOLO_PT_PATH", PT_PATH),
        help=f"YOLO .pt model path (default: {PT_PATH}).",
    )
    parser.add_argument(
        "--data",
        default=os.environ.get("VCTS_CALIB_DATA_YAML", DATA_YAML),
        help=f"Calibration dataset YAML path (default: {DATA_YAML}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.export(
        format="tflite",
        imgsz=640,
        int8=True,
        data=args.data,
    )
    print("TFLite INT8 export completed. Check the export folder for a .tflite file.")


if __name__ == "__main__":
    main()
