import logging
import math
import cv2

logger = logging.getLogger(__name__)


def draw_overlay_cv(frame, boxes, tracked, crossings,
                    line_start, line_end,
                    display_w, display_h, in_w, in_h,
                    tile_bounds=None):
    """
    Draw detection boxes, tracked centroids, the crossing line, and count
    onto a frame using OpenCV.

    Args:
        frame: BGR numpy array to draw on (mutated in-place).
        boxes: list of [x1, y1, x2, y2] in display coordinates.
        tracked: dict of {objectID: (cx, cy)} in ML coordinates.
        crossings: int – current crossing count.
        line_start: (x, y) tuple in display coordinates.
        line_end: (x, y) tuple in display coordinates.
        display_w, display_h: display resolution.
        in_w, in_h: model input resolution (for centroid scaling).
        tile_bounds: optional (x0, y0, x1, y1) to draw the tile ROI rectangle.

    Returns:
        The annotated frame.
    """
    if tile_bounds is not None:
        tx0, ty0, tx1, ty1 = tile_bounds
        cv2.rectangle(frame, (tx0, ty0), (tx1, ty1), (128, 128, 255), 2)

    cv2.line(frame, line_start, line_end, (0, 255, 255), 3)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    scale_x = display_w / in_w
    scale_y = display_h / in_h
    for objectID, centroid in tracked.items():
        cx = int(centroid[0] * scale_x)
        cy = int(centroid[1] * scale_y)
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"ID{objectID}",
            (cx - 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        frame,
        f"Crossings: {crossings}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
    return frame


def draw_overlay_cairo(ctx, boxes, tracked, crossings,
                       line_start, line_end,
                       display_w, display_h, in_w, in_h,
                       tile_bounds=None):
    """
    Draw detection boxes, tracked centroids, the crossing line, and count
    using a Cairo context (for GStreamer cairooverlay).

    Args:
        ctx: Cairo drawing context.
        boxes: list of [x1, y1, x2, y2] in display coordinates.
        tracked: dict of {objectID: (cx, cy)} in ML coordinates.
        crossings: int – current crossing count.
        line_start: (x, y) tuple in display coordinates.
        line_end: (x, y) tuple in display coordinates.
        display_w, display_h: display resolution.
        in_w, in_h: model input resolution (for centroid scaling).
        tile_bounds: optional (x0, y0, x1, y1) to draw the tile ROI rectangle.
    """
    try:
        import cairo as _cairo
        _FONT_SLANT_NORMAL = _cairo.FONT_SLANT_NORMAL
        _FONT_WEIGHT_NORMAL = _cairo.FONT_WEIGHT_NORMAL
    except ImportError:
        _FONT_SLANT_NORMAL = 0
        _FONT_WEIGHT_NORMAL = 0

    scale_x = display_w / in_w
    scale_y = display_h / in_h

    if tile_bounds is not None:
        tx0, ty0, tx1, ty1 = tile_bounds
        ctx.set_source_rgb(1.0, 0.5, 0.5)
        ctx.set_line_width(2)
        ctx.rectangle(tx0, ty0, tx1 - tx0, ty1 - ty0)
        ctx.stroke()

    ctx.set_source_rgb(1.0, 1.0, 0.0)
    ctx.set_line_width(3)
    ctx.move_to(line_start[0], line_start[1])
    ctx.line_to(line_end[0], line_end[1])
    ctx.stroke()

    ctx.set_source_rgb(0.0, 0.0, 1.0)
    ctx.set_line_width(2)
    for box in boxes:
        x1, y1, x2, y2 = box
        ctx.rectangle(x1, y1, x2 - x1, y2 - y1)
        ctx.stroke()

    for objectID, centroid in tracked.items():
        cx = int(centroid[0] * scale_x)
        cy = int(centroid[1] * scale_y)
        ctx.set_source_rgb(0.0, 1.0, 0.0)
        ctx.arc(cx, cy, 8, 0, 2 * math.pi)
        ctx.fill()

        ctx.select_font_face("Sans", _FONT_SLANT_NORMAL, _FONT_WEIGHT_NORMAL)
        ctx.set_font_size(16)
        ctx.move_to(cx - 10, cy - 10)
        ctx.show_text(f"ID{objectID}")

    ctx.set_source_rgb(1.0, 1.0, 0.0)
    ctx.set_font_size(32)
    ctx.move_to(10, 40)
    ctx.show_text(f"Crossings: {crossings}")
