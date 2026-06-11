def detect_persons(
    model,
    frame,
    confidence_threshold=0.3,
    min_area=500,
    imgsz=320,
    device=None,
    half=False,
):
    """Run YOLO tracking on a single frame and return person detections.

    Args:
        model: Loaded YOLO model (ultralytics.YOLO).
        frame: OpenCV frame (numpy array).
        confidence_threshold: Minimum confidence to keep a detection.
        min_area: Minimum bounding box area (in pixels) to keep a detection.
    Returns:
        List of dicts with keys: 'box', 'conf', 'class_id', 'centroid',
        'foot_point', and 'track_id'.
    """
    # Track only people at inference time. Filtering before post-processing keeps
    # IDs focused on the class that matters for line crossing.
    results = model.track(
        frame,
        persist=True,
        verbose=False,
        classes=[0],
        conf=confidence_threshold,
        imgsz=imgsz,
        device=device,
        half=half,
    )
    result = results[0]
    person_detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            if class_id != 0 or conf < confidence_threshold:
                continue
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = xyxy
            # Filter out tiny boxes (likely false positives)
            if (x2 - x1) * (y2 - y1) < min_area:
                continue
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)
            track_id = int(box.id[0]) if box.id is not None else None
            person_detections.append({
                'box': xyxy,
                'conf': conf,
                'class_id': class_id,
                'centroid': (cx, cy),
                'foot_point': (foot_x, foot_y),
                'track_id': track_id,
            })
    return person_detections
