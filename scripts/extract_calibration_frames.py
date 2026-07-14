import argparse
import os
from pathlib import Path

import cv2

DEFAULT_CALIBRATION_IMAGES_DIR = "C:/Users/saath/VCTS_DATA/calibration_images"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract representative calibration frames from a video."
    )
    parser.add_argument(
        "--source",
        default="The CCTV People Demo 2_1080p.mp4",
        help="Input video path.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("VCTS_CALIBRATION_IMAGES_DIR", DEFAULT_CALIBRATION_IMAGES_DIR),
        help="Output folder for extracted frames.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Approximate frames to save per second of video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Maximum number of calibration frames to save.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.source}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(int(round(source_fps / max(args.fps, 0.001))), 1)

    saved = 0
    frame_idx = 0
    while saved < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % stride == 0:
            out_path = out_dir / f"frame_{saved:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"Saved {saved} calibration frames to {out_dir}")


if __name__ == "__main__":
    main()
