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
from pathlib import Path

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_SOURCE = "video.mp4"
YOLOV8N_MODEL_NAME = "yolov8n.pt"


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
        default=640,
        help="Target frame width after resizing (default 640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=360,
        help="Target frame height after resizing (default 360).",
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
        default=1,
        help="Process one out of every N frames (default 1, best accuracy).",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for person detections (default 0.5).",
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
    parser.add_argument(
        "--line-margin",
        type=float,
        default=3.0,
        help="Pixel dead zone around the line to reduce jitter counts (default 3).",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=1,
        help="Consecutive frames required before accepting a side change (default 1).",
    )
    parser.add_argument(
        "--crossing-cooldown",
        type=int,
        default=8,
        help="Processed frames to wait before the same track can count again (default 8).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-frame detection details.",
    )
    return parser.parse_args()


def validate_yolov8n_only(model_path):
    if Path(model_path).name.lower() != YOLOV8N_MODEL_NAME:
        raise SystemExit(
            f"This project is locked to {YOLOV8N_MODEL_NAME}. "
            f"Received: {model_path}"
        )


def get_boundary_points(width, height):
    start_x_ratio = 700 / 1920
    start_y_ratio = 500 / 1080
    end_x_ratio = 1600 / 1920
    end_y_ratio = 0.6
    start = (int(width * start_x_ratio), int(height * start_y_ratio))
    end = (int(width * end_x_ratio), int(height * end_y_ratio))
    return start, end


def signed_line_distance(point, line_start, line_end):
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    dx = x2 - x1
    dy = y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    return ((px - x1) * dy - (py - y1) * dx) / length


def side_from_distance(distance, margin):
    if distance > margin:
        return "side_a"
    if distance < -margin:
        return "side_b"
    return None


def point_orientation(a, b, c):
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(value) < 1e-6:
        return 0
    return 1 if value > 0 else 2


def point_on_segment(a, b, c):
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def segments_intersect(a, b, c, d):
    o1 = point_orientation(a, b, c)
    o2 = point_orientation(a, b, d)
    o3 = point_orientation(c, d, a)
    o4 = point_orientation(c, d, b)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and point_on_segment(a, c, b):
        return True
    if o2 == 0 and point_on_segment(a, d, b):
        return True
    if o3 == 0 and point_on_segment(c, a, d):
        return True
    if o4 == 0 and point_on_segment(c, b, d):
        return True
    return False


def crossed_boundary(prev_point, cur_point, line_start, line_end, prev_distance, cur_distance):
    if prev_point is None:
        return False
    if prev_distance == 0 or cur_distance == 0:
        return segments_intersect(prev_point, cur_point, line_start, line_end)
    sign_changed = (prev_distance > 0) != (cur_distance > 0)
    return sign_changed and segments_intersect(prev_point, cur_point, line_start, line_end)


def update_track_side(track_state, raw_side, frame_count, stable_frames):
    if raw_side is None:
        return None

    if raw_side == track_state.get("pending_side"):
        track_state["pending_count"] = track_state.get("pending_count", 0) + 1
    else:
        track_state["pending_side"] = raw_side
        track_state["pending_count"] = 1

    track_state["last_seen"] = frame_count
    if track_state["pending_count"] >= stable_frames:
        return raw_side
    return None



def main():
    args = parse_args()
    validate_yolov8n_only(args.model)
    YOLO = load_yolo()
    out_writer = None

    print(f"Loading YOLOv8 model: {args.model}")
    model = YOLO(args.model)
    device = "cpu"
    half = False
    # Move model to GPU only if --use-gpu flag is passed and CUDA is available
    if args.use_gpu and torch.cuda.is_available():
        try:
            device = "cuda:0"
            half = True
            model.to(device)
            print("Using CUDA GPU for inference (forced).")
        except Exception as e:
            device = "cpu"
            half = False
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

        line_start, line_end = get_boundary_points(args.width, args.height)
        print(
            f"Video Dimensions: {args.width}x{args.height}. "
            f"Virtual line: {line_start} -> {line_end}"
        )
        
        # Initialize crossing count and previous side
        crossing_count = 0
        # Initialize speed factor for display playback (e.g., 2× faster)
        speed_factor = 1.0  # used for waitKey timing
        # Optional video writer (output.mp4) if user wants to save
        # Optional video writer (output.mp4) if user wants to save
        save_output = args.save_output
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_fps = max((fps or 30.0) / max(args.skip_frames, 1), 1.0)
            out_writer = cv2.VideoWriter("output.mp4", fourcc, output_fps * speed_factor, (args.width, args.height))

        # Per-track state makes crossing counts depend on stable tracked motion.
        track_states = {}

            

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
                frame_idx += 1
                draw_custom_line(frame, line_start, line_end, color=(0, 255, 255), thickness=3)

                # Run detection on every frame (no motion filter)
                detections = detect_persons(
                    model,
                    frame,
                    confidence_threshold=args.conf_thresh,
                    imgsz=max(args.width, args.height),
                    device=device,
                    half=half,
                )

                if args.debug and detections:
                    print(f"Detections this frame: {len(detections)}")
                
                # Draw detections if any are present
                if len(detections) > 0:
                    frame = draw_detections(frame, detections)

                # Per-object crossing detection using tracked foot points.
                for det in detections:
                    tid = det.get('track_id')
                    if tid is None:
                        continue

                    state = track_states.setdefault(
                        tid,
                        {
                            "stable_side": None,
                            "pending_side": None,
                            "pending_count": 0,
                            "last_count_frame": -10_000,
                            "last_seen": frame_count,
                            "last_foot_point": None,
                            "last_distance": None,
                        },
                    )
                    state["last_seen"] = frame_count
                    foot_point = det["foot_point"]
                    distance = signed_line_distance(foot_point, line_start, line_end)
                    raw_side = side_from_distance(distance, args.line_margin)
                    stable_side = update_track_side(
                        state,
                        raw_side,
                        frame_count,
                        max(args.stable_frames, 1),
                    )

                    previous_side = state.get("stable_side")
                    path_crossed = crossed_boundary(
                        state.get("last_foot_point"),
                        foot_point,
                        line_start,
                        line_end,
                        state.get("last_distance", distance),
                        distance,
                    )
                    can_count = (
                        (
                            path_crossed
                            or (
                                stable_side is not None
                                and previous_side is not None
                                and stable_side != previous_side
                            )
                        )
                        and frame_count - state["last_count_frame"] >= args.crossing_cooldown
                    )
                    if can_count:
                        crossing_count += 1
                        state["last_count_frame"] = frame_count
                        print(f"[ALERT] Track {tid} crossed. Total: {crossing_count}")
                    if stable_side is not None:
                        state["stable_side"] = stable_side
                    if raw_side is not None:
                        state["last_foot_point"] = foot_point
                        state["last_distance"] = distance

                stale_tracks = [
                    tid for tid, state in track_states.items()
                    if frame_count - state.get("last_seen", frame_count) > 120
                ]
                for tid in stale_tracks:
                    del track_states[tid]

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

                # Inference already controls throughput; keep display wait minimal.
                if cv2.waitKey(1) & 0xFF == ord('q'):
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
