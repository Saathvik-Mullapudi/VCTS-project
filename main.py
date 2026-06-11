# main.py
# Entry point for the line-crossing demo.
# Loads YOLO model, processes video frames, draws a manually adjustable foot-path line,
# runs person detection, tracks crossing events, and optionally saves output video.
import argparse
import cv2
import sys
import torch
from visualizer import draw_detections, draw_custom_line
import time
from detector import detect_persons
import math

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_SOURCE = "video.mp4"


def load_yolo():
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics\n"
            "Install the project dependencies with:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc

    return YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a YOLOv8 detection smoke test on an image or video source."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Target frame width after resizing (default 320).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=240,
        help="Target frame height after resizing (default 240).",
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save processed video to output.mp4",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"YOLO model path or name. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=2,
        help="Process one out of every N frames (default 2) to increase speed.",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.3,
        help="Confidence threshold for person detections (default 0.3).",
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Image, video, webcam index, or URL to run inference on.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Force inference on CUDA GPU if available.",
    )
    return parser.parse_args()



def main():
    args = parse_args()
    YOLO = load_yolo()

    print(f"Loading YOLOv8 model: {args.model}")
    model = YOLO(args.model)
    # Move model to GPU only if --use-gpu flag is passed and CUDA is available
    if args.use_gpu and torch.cuda.is_available():
        try:
            model.to('cuda')
            print("Using CUDA GPU for inference (forced).")
        except Exception as e:
            print(f"CUDA error ({e}), falling back to CPU.")
    elif args.use_gpu:
        print("--use-gpu passed but CUDA not available, running on CPU.")
    else:
        print("Running on CPU.")


    # Determine if source is an image
    is_image = False
    source_str = str(args.source)
    ext = source_str.lower().split('.')[-1]
    if ext in ['jpg', 'jpeg', 'png', 'webp', 'bmp']:
        is_image = True

    if is_image:
        print(f"Running inference on image source: {args.source}")
        results = model(args.source)
        for r in results:
            print("\nDetection Summary:")
            print(f"Detected {len(r.boxes)} objects.")
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                print(f" - Class: {label}, Confidence: {conf:.2f}")
    else:
        # Step 4: Open video file
        print(f"Opening video file: {args.source}")
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            print(f"Error: Could not open video file or source: {args.source}")
            sys.exit(1)

        # Determine virtual line position (60% down the frame)
        line_y = int(args.height * 0.6)
        print(f"Video Dimensions: {args.width}x{args.height}. Virtual line set at Y = {line_y}")
        
        # Initialize crossing count and previous side
        crossing_count = 0
        previous_side = None
        # Initialize speed factor for display playback (e.g., 2× faster)
        speed_factor = 1.0  # used for waitKey timing
        # Optional video writer (output.mp4) if user wants to save
        # Optional video writer (output.mp4) if user wants to save
        save_output = args.save_output
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter("output.mp4", fourcc, fps * speed_factor, (args.width, args.height))
        else:
            out_writer = None

        # Dictionary to keep previous side per tracked object
        prev_sides = {}

            

        print("Video opened successfully. Starting frame processing loop...")
        
        try:

            frame_count = 0
            start_time = time.time()

            frame_idx = 0
            while True:
                # Read next frame from video
                ret, frame = cap.read()
                if not ret:
                    print("Reached end of video or failed to read frame.")
                    break
                # Skip frames based on user setting to speed up processing
                if frame_idx % args.skip_frames != 0:
                    frame_idx += 1
                    continue
                # Resize frame to target dimensions
                frame = cv2.resize(frame, (args.width, args.height))
                # Compute scaled line coordinates
                start_x_ratio = 700 / 1920
                start_y_ratio = 500 / 1080
                end_x_ratio = 1600 / 1920
                START_X = int(args.width * start_x_ratio)
                START_Y = int(args.height * start_y_ratio)
                END_X = int(args.width * end_x_ratio)
                END_Y = line_y
                frame_idx += 1
                draw_custom_line(frame, (START_X, START_Y), (END_X, END_Y), color=(0, 255, 255), thickness=3)

                # Run detection on every frame (no motion filter)
                detections = detect_persons(model, frame, confidence_threshold=args.conf_thresh)

                if detections:
                    print(f"Detections this frame: {len(detections)}")
                
                # Draw detections if any are present
                if len(detections) > 0:
                    frame = draw_detections(frame, detections)

                # Per-object crossing detection using track IDs
                any_tracked = False
                for det in detections:
                    tid = det.get('track_id')
                    if tid is None:
                        continue
                    any_tracked = True
                    side = "above" if det['centroid'][1] < line_y else "below"
                    prev_side = prev_sides.get(tid)
                    if prev_side is not None and side != prev_side:
                        crossing_count += 1
                        print(f"[ALERT] Crossing detected. Total: {crossing_count}")
                    prev_sides[tid] = side
                # Fallback majority-side detection when no tracking IDs are present
                if not any_tracked:
                    above = sum(1 for det in detections if det['centroid'][1] < line_y)
                    below = sum(1 for det in detections if det['centroid'][1] >= line_y)
                    current_side = None
                    if above > below:
                        current_side = "above"
                    elif below > above:
                        current_side = "below"
                    if current_side and previous_side and current_side != previous_side:
                        crossing_count += 1
                        print(f"[ALERT] Crossing detected (fallback). Total: {crossing_count}")
                    if current_side:
                        previous_side = current_side

                # Increment frame counter for the next iteration
                frame_count += 1
                # No need to reset prev_detections; tracking handled via prev_sides dict

                # Overlay crossing count on the frame
                cv2.putText(
                    frame,
                    f"Crossings: {crossing_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                )

                # Write frame to output video if enabled
                if out_writer is not None:
                    out_writer.write(frame)
                
                # Step 6: Display frames
                cv2.imshow("Video Playback (Press 'q' to quit)", frame)

                # Dynamic wait based on source FPS to keep real-time playback
                # Adjust wait time according to speed_factor for faster display
                frame_time_ms = int(1000 / ((fps or 30) * speed_factor))
                if cv2.waitKey(frame_time_ms) & 0xFF == ord('q'):
                    print("Playback interrupted by user.")
                    break
            print(f"Finished processing {frame_count} frames.")
            elapsed = time.time() - start_time
            fps_calc = frame_count / elapsed if elapsed > 0 else 0
            print(f"Processed {frame_count} frames in {elapsed:.2f}s -> {fps_calc:.2f} FPS")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    # Release video writer if saving output
    if out_writer is not None:
        out_writer.release()
        print("Saved processed video to output.mp4")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
