# Line Crossing Detection with YOLOv8n

Fast, readable people-crossing detection built with **OpenCV** and **YOLOv8n**.  
The app watches a video, tracks people, draws a virtual boundary, and counts when a tracked person moves from one side of the line to the other.

> Model rule: this project is intended to stay on `yolov8n.pt` for speed. No larger model is required.

## At A Glance

| Area | Current Behavior |
| --- | --- |
| Model | `yolov8n.pt` by default |
| Input | Image, video file, webcam index, or stream URL |
| Detection | YOLO tracking with person filtering |
| Speed controls | Resize + `--skip-frames` |
| Output | Live OpenCV window, optional `output.mp4` |
| Main sample | `video.mp4` |

## Project Map

```text
line_crossing/
|-- main.py          # CLI, video loop, line placement, crossing count
|-- detector.py      # YOLO tracking wrapper, person/confidence filtering
|-- visualizer.py    # Boundary line, boxes, labels, centroid drawing
|-- motion_utils.py  # Motion-detection helper functions, not yet wired in
|-- requirements.txt # opencv-python, ultralytics, numpy
|-- yolov8n.pt       # lightweight model used by default
|-- video.mp4        # default sample video
|-- bus.jpg          # sample image
```

## Pipeline

```mermaid
flowchart TD
    A[main.py] --> B[Parse CLI args]
    B --> C[Load YOLOv8n]
    C --> D{Input type}
    D -->|Image| E[Run inference]
    E --> F[Print detections]
    D -->|Video| G[Open VideoCapture]
    G --> H[Read frame]
    H --> I{Frame ok?}
    I -->|No| Z[Release resources]
    I -->|Yes| J{Skip frame?}
    J -->|Yes| H
    J -->|No| K[Resize]
    K --> L[Draw boundary]
    L --> M[Track people]
    M --> N[Filter class 0: person]
    N --> O[Draw overlays]
    O --> P[Update crossing count]
    P --> Q[Display / save]
    Q --> H
```

## Crossing Logic

```mermaid
stateDiagram-v2
    [*] --> NewTrack
    NewTrack --> Above: centroid above line_y
    NewTrack --> Below: centroid below line_y
    Above --> Above: same side
    Below --> Below: same side
    Above --> Counted: side changed
    Below --> Counted: side changed
    Counted --> Above
    Counted --> Below
```

The visual boundary is drawn as a slanted custom line, but the current crossing check still uses horizontal `line_y`. That is the biggest accuracy upgrade waiting to happen.

## Setup

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
python main.py
python main.py --model yolov8n.pt
python main.py --use-gpu --model yolov8n.pt
python main.py --save-output --model yolov8n.pt
python main.py --width 320 --height 240 --skip-frames 2 --conf-thresh 0.3
```

Press `q` in the playback window to stop.

## CLI Options

| Option | Default | Meaning |
| --- | ---: | --- |
| `--source` | `video.mp4` | Image, video, webcam, or URL |
| `--model` | `yolov8n.pt` | YOLO model path |
| `--width` | `320` | Frame width after resize |
| `--height` | `240` | Frame height after resize |
| `--skip-frames` | `2` | Process 1 of every N frames |
| `--conf-thresh` | `0.3` | Minimum person confidence |
| `--save-output` | off | Save annotated `output.mp4` |
| `--use-gpu` | off | Move model to CUDA if available |

## Next Improvements

1. Enforce YOLOv8n-only and remove the heavier model from the workflow.
2. Auto-select CUDA and pass faster YOLO args: `classes=[0]`, `conf`, `imgsz`, `device`.
3. Count against the actual slanted line using bottom-center foot points.
4. Add per-track debounce to avoid repeat counts near the line.
5. Wire `motion_utils.py` into the loop to skip low-motion frames.
