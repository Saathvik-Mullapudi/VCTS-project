# motion_utils.py
"""Utility functions for lightweight motion detection used to skip YOLO inference on low‑motion frames.

The implementation uses simple frame differencing with a configurable mean‑absolute‑difference threshold.
"""
import cv2
import numpy as np

def init_motion_state(frame):
    """Initialize motion detection state from the first video frame.

    Returns a grayscale version of the provided frame which will be used as the
    reference for subsequent frame‑to‑frame comparisons.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def has_significant_motion(prev_gray, cur_frame, thresh):
    """Determine whether the current frame has significant motion.

    Parameters
    ----------
    prev_gray : np.ndarray
        Grayscale image of the previous frame.
    cur_frame : np.ndarray
        Grayscale image of the current frame.
    thresh : float
        Mean absolute pixel difference threshold. If the computed mean is greater
        than ``thresh`` the frame is considered to have motion.

    Returns
    -------
    (bool, np.ndarray)
        A tuple ``(significant, diff)`` where ``significant`` indicates whether
        motion exceeds the threshold and ``diff`` contains the absolute
        difference image (useful for debugging or visualisation).
    """
    # Compute absolute difference between frames
    diff = cv2.absdiff(prev_gray, cur_frame)
    # Mean absolute difference per pixel
    mean_diff = np.mean(diff)
    return mean_diff > thresh, diff
