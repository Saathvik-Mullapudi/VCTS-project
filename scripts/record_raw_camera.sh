#!/bin/bash
# ============================================================================
# RAW CAMERA RECORDING SCRIPT FOR i.MX8 (Task 1.3)
# ============================================================================
# This script records exactly 60 seconds of raw video from the IMX462 camera
# and saves it to an MP4 file. 
# 
# WHY DO WE NEED THIS?
# When you lose access to the physical board and the camera in 5 days, you 
# will need this video to test and tune the Line Crossing math on your laptop.
# ============================================================================

CAMERA_DEVICE="/dev/video3"
OUTPUT_FILE="raw_calibration_video.mp4"
DURATION_SECONDS=60
FRAMERATE=30

# Calculate total frames to record
TOTAL_FRAMES=$((DURATION_SECONDS * FRAMERATE))

echo "Starting recording from $CAMERA_DEVICE..."
echo "This will take $DURATION_SECONDS seconds. Please walk across the camera view!"

# GStreamer pipeline using NXP i.MX8 hardware encoder (vpuenc_h264)
gst-launch-1.0 -e \
    v4l2src device=$CAMERA_DEVICE num-buffers=$TOTAL_FRAMES ! \
    video/x-raw,framerate=$FRAMERATE/1 ! \
    imxvideoconvert_g2d ! videoconvert ! video/x-raw,format=I420 ! \
    vpuenc_h264 bitrate=5000 ! \
    h264parse ! \
    mp4mux ! \
    filesink location=$OUTPUT_FILE

echo "Recording complete! Saved to $OUTPUT_FILE"
echo "IMPORTANT: Copy $OUTPUT_FILE to your laptop so you have it forever."
