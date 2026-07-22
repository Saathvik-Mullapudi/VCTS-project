# Line Crossing Smoke Test Checklist

Before committing any structural or logic changes to the repository, ensure the application passes the following smoke tests:

## 1. Startup
- [ ] Application launches without syntax or import errors
- [ ] Configurations (`configs/settings.py`) load successfully
- [ ] Model (TFLite/TensorFlow) loads successfully
- [ ] Hardware acceleration initializes (if available)

## 2. Pipeline
- [ ] GStreamer/OpenCV pipeline starts correctly
- [ ] Frames are successfully received and decoded
- [ ] No pipeline errors or crashes in GStreamer bus

## 3. Inference
- [ ] Persons are successfully detected in the frame
- [ ] Bounding boxes are accurately mapped and drawn
- [ ] Confidence values appear reasonable (> `CONF_THRESHOLD`)

## 4. Tracking & Line Crossing
- [ ] Objects are tracked consistently across frames
- [ ] Line crossing count increments correctly when a person crosses the line
- [ ] No duplicate counts for the same person loitering near the line

## 5. Output
- [ ] Video output file is saved correctly (`outputs/videos/output.mp4`)
- [ ] Output video is playable and contains overlays
- [ ] Metrics are generated (`outputs/metrics/metrics.csv` and `.json`)
- [ ] Logs are written to `outputs/logs/line_crossing.log`

## 6. Shutdown
- [ ] Exits cleanly on end-of-stream (EOS) or `Ctrl+C`
- [ ] Resources (cameras, video writers, threads) are gracefully released
- [ ] No segmentation faults or zombie processes
