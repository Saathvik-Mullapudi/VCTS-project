# visualizer.py
# ------------------------------------------------------------------
# Utility functions for drawing on video frames.
# Includes:
#   - draw_virtual_line: full‑width line (kept for compatibility)
#   - draw_custom_line: manual line with user‑specified start/end points
#   - draw_detections: bounding boxes, foot points, track IDs and confidence labels
# ------------------------------------------------------------------
import cv2

def draw_virtual_line(frame, line_y, color=(0, 0, 255), thickness=3):
    """
    Draws a horizontal virtual crossing line across the entire frame.
    
    Args:
        frame: The video frame (numpy array) to draw on.
        line_y: The Y-coordinate of the horizontal line.
        color: The BGR color of the line (default is red: (0, 0, 255)).
        thickness: The thickness of the line in pixels.
    """
    # frame.shape returns a tuple (height, width, channels)
    # We retrieve the width of the frame so we can draw from x=0 to x=width
    width = frame.shape[1]
    
    # cv2.line draws a line on 'frame' from point (0, line_y) to (width, line_y)
    # - Point 1: (0, line_y) represents the left border
    # - Point 2: (width, line_y) represents the right border
    # - Color: BGR format (default red: (0, 0, 255))
    # - Thickness: line stroke thickness in pixels
    cv2.line(frame, (0, line_y), (width, line_y), color, thickness)
    
    # cv2.putText adds visual helper text to the frame
    # - (10, line_y - 10) puts the text 10 pixels to the right and 10 pixels above the line
    # - FONT_HERSHEY_SIMPLEX is a standard OpenCV font
    # - 0.6 is the font scale (size multiplier)
    # - 2 is the font thickness
    cv2.putText(
        frame, 
        "Virtual Crossing Boundary", 
        (10, line_y - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, 
        color, 
        2
    )
    return frame
def draw_custom_line(frame,
                     pt_start: tuple[int, int],
                     pt_end:   tuple[int, int],
                     color:    tuple[int, int, int] = (0, 255, 255),
                     thickness: int = 3) -> None:
    """
    Draw a line between two manually‑specified points.
    """
    cv2.line(frame, pt_start, pt_end, color, thickness)

    # optional label centred on the line
    mid_x = (pt_start[0] + pt_end[0]) // 2
    mid_y = (pt_start[1] + pt_end[1]) // 2
    cv2.putText(frame,
                "Virtual Crossing Boundary",
                (mid_x - 60, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2)


def draw_detections(frame, detections):
    """
    Draws bounding boxes, tracking IDs, centroids, and foot points for detections.
    
    Args:
        frame: The OpenCV frame (numpy array) to draw on.
        detections: List of detection dictionaries containing 'box', 'conf',
        'centroid', 'foot_point', and 'track_id'.
        
    Returns:
        The frame with visual elements drawn on it.
    """
    for det in detections:
        box = det['box']
        conf = det['conf']
        cx, cy = det['centroid']
        fx, fy = det.get('foot_point', (cx, cy))
        track_id = det.get('track_id')
        # Default colors (no tracking, no crossing alerts)
        box_color = (255, 0, 0)  # Blue in BGR
        dot_color = (0, 255, 0)  # Green in BGR
        # Bounding box corners as integer coordinates
        x1, y1, x2, y2 = map(int, box)

        # Draw bounding box rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Draw centroid dot
        cv2.circle(frame, (cx, cy), 5, dot_color, -1)

        # Draw bottom-center foot point used for line-crossing decisions
        cv2.circle(frame, (fx, fy), 5, (0, 255, 255), -1)
        
        label = f"ID {track_id} | {conf:.2f}" if track_id is not None else f"Conf: {conf:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            2,
        )      
    return frame
