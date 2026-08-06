import sys
import os

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
except ImportError:
    print("Installing python-pptx library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-pptx"])
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor

def create_deck():
    prs = Presentation()
    # Set slide dimensions to widescreen 16:9
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]

    slides_data = [
        {
            "title": "Edge-AI Line Crossing Detector on NXP i.MX8",
            "subtitle": "Porting, Hardware Offloading, and Optimization of Real-Time Vision Systems",
            "bullets": [
                "Presenter: [Your Name]",
                "Target Hardware: NXP i.MX8M Plus SoC (VIP8000 NPU)",
                "Focus: Edge ML Inference, GStreamer Hardware Pipelines, Self-Healing Resilience"
            ],
            "notes": "Welcome everyone. Today I'll walk through the porting, hardware offloading, and performance optimization of our live line-crossing vision system on the NXP i.MX8 edge platform."
        },
        {
            "title": "Initial Prototype & Performance Bottlenecks",
            "subtitle": "Transitioning from Laptop Prototyping to Resource-Constrained Edge Chips",
            "bullets": [
                "Initial Prototype: OpenCV Python capture (cv2.VideoCapture) running on a laptop.",
                "Edge Constraints: Porting to i.MX8 caused CPU usage to hit 85%+ with infinite video lag.",
                "Initial Attempt (Manual Frame Skipping): Skips every 3rd/5th frame in Python.",
                "Limitations: Rigid skipping failed when camera framerates or NPU timing fluctuated."
            ],
            "notes": "When I first ported our laptop computer vision prototype to the i.MX8 edge board, CPU usage reached 85%, and the video stream accumulated lag. My first step was writing a manual frame-skipper in Python. However, fixed frame-skipping proved too rigid to handle dynamic camera framerates and inference timing variations."
        },
        {
            "title": "Model Quantization & NPU Hardware Delegation",
            "subtitle": "Offloading Neural Network Matrix Operations from CPU to VIP8000 NPU",
            "bullets": [
                "Float32 CPU Bottleneck: Standard float32 YOLOv8n execution on CPU took > 400 ms per frame.",
                "Full-Integer Quantization: Converted model to 8-bit full-integer TFLite (uint8/int8).",
                "NPU Driver Delegate: Configured TFLite runtime to load NXP libvx_delegate.so driver.",
                "Performance Impact: Latency dropped to 78.2 ms, holding average NPU load at 21.5%."
            ],
            "notes": "To reduce inference latency, I quantized YOLOv8n to 8-bit full-integer TFLite format. I then configured the TFLite runtime to load NXP's libvx_delegate.so driver, offloading tensor calculations directly to the onboard VIP8000 NPU. This reduced inference latency to 78 milliseconds while using 21.5% of the NPU capacity."
        },
        {
            "title": "Pipeline Refinement & Dynamic Frame Dropping",
            "subtitle": "Offloading Colorspace Conversion to GPU & Replacing Manual Frame Skipping",
            "bullets": [
                "GPU Colorspace Scaling: Used imxvideoconvert_g2d to offload YUV-to-BGRx scaling to 2D GPU.",
                "Replaced Manual Skipping: Configured GStreamer appsink with drop=true and max-buffers=1.",
                "Decoupled Architecture: AppSink dynamically drops stale intermediate frames while NPU is busy.",
                "Result: Visual streaming rate (11.2 FPS) decoupled from NPU inference rate (6.0 FPS)."
            ],
            "notes": "After offloading model execution, I turned to pipeline optimization. First, I offloaded colorspace conversion and scaling to the i.MX8 GPU using GStreamer's imxvideoconvert_g2d element. Next, I replaced the manual frame-skipping logic with GStreamer's appsink element configured with drop=true and max-buffers=1. This dynamically drops stale frames at the buffer boundary, guaranteeing zero video lag while keeping video streaming smooth at 11.2 FPS."
        },
        {
            "title": "Boundary Math & False-Positive Elimination",
            "subtitle": "Foot-Point Vectors, EMA Centroid Smoothing, Spatial Padding, and Debouncing",
            "bullets": [
                "Foot-Point Tracking: Tracked bottom-center point (x + w/2, y2) to stabilize floor contact.",
                "EMA Centroid Smoothing: Applied smoothing (alpha=0.75) to eliminate bounding box jitter.",
                "Vector Cross-Product: Computed 2D signed cross product d(P) = (x - x1)(y2 - y1) - (y - y1)(x2 - x1).",
                "Spatial & Temporal Filters: 30px spatial buffer padding + 2-frame debouncing state machine."
            ],
            "notes": "To ensure count accuracy, I addressed bounding box jitter near the line. I updated the math to track foot points instead of box centers, applied Exponential Moving Average smoothing to centroids, and added a 30-pixel spatial padding zone with a 2-frame debouncing state machine. This eliminated false counts caused by loitering individuals."
        },
        {
            "title": "Industrial Resilience & Fault Tolerance",
            "subtitle": "Handling Hardware Disconnects and Process Recovery Gracefully",
            "bullets": [
                "Hardware Disconnect Recovery: Catches Gst.MessageType.ERROR when camera is unplugged.",
                "Hardware Lock Cleanup: Transitions GStreamer to Gst.State.NULL to release /dev/video3 locks.",
                "Stepped Backoff: Retries every 3s for 2 minutes (40 retries), then backs off to 3m intervals.",
                "OS Service Integration: Registered GLib.unix_signal_add(SIGTERM) for 3s systemd auto-recovery."
            ],
            "notes": "For field deployments, I implemented a self-healing layer. If a camera cable is unplugged, our GStreamer bus callback catches the error, releases hardware device locks, and enters a stepped-backoff reconnect loop. If the process is terminated, systemd automatically restarts the service within 3 seconds."
        },
        {
            "title": "Benchmarking & System Verification (30.7-Minute Run)",
            "subtitle": "Continuous Stress Testing on Live Camera Streams",
            "bullets": [
                "Test Duration: Processed 26,911 frames continuously over 30.7 minutes.",
                "Visual & Inference Rates: RTSP Output at 11.2 FPS | NPU Inference at 6.0 FPS.",
                "Hardware Loads: NPU Avg 21.5% (Max 42%) | CPU Avg 47.3% (Max 84%).",
                "Memory & Thermal Stability: RAM flat at 18.9% (zero leaks) | Temperature stable at 80.6°C."
            ],
            "notes": "I verified system performance with a 30-minute stress test on live camera feeds. The benchmark results confirmed stability: RAM usage remained flat at 18.9% with zero memory leaks, NPU load averaged 21.5%, and CPU temperature stabilized safely at 80.6°C."
        },
        {
            "title": "Key Takeaways & Systems Engineering Learnings",
            "subtitle": "Reflections on Hardware-Software Co-Design",
            "bullets": [
                "Hardware Profiling First: Measured sysfs and debugfs (/sys/kernel/debug/gc/load) drivers.",
                "Iterative Architecture: Replaced naive frame-skipping with dynamic appsink ring buffers.",
                "Production Readiness: Designed self-healing code for OS signals, hardware unplugs, and locks.",
                "Engineering Growth: Transitioned from writing scripts to building edge ML vision systems."
            ],
            "notes": "Reflecting on this project, my biggest learning was approaching software from a systems perspective—profiling low-level hardware registers, iterating on pipeline architecture, and building resilient code designed for real-world hardware."
        }
    ]

    for data in slides_data:
        slide = prs.slides.add_slide(blank_layout)
        
        # Add Header Background banner
        header = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(1.3)) # 1 = MSO_SHAPE.RECTANGLE
        header.fill.solid()
        header.fill.fore_color.rgb = RGBColor(15, 32, 67) # Dark Blue
        header.line.color.rgb = RGBColor(15, 32, 67)

        # Title text
        tf = header.text_frame
        tf.margin_left = Inches(0.8)
        tf.margin_top = Inches(0.2)
        p = tf.paragraphs[0]
        p.text = data["title"]
        p.font.size = Pt(28)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)
        
        if data["subtitle"]:
            p2 = tf.add_paragraph()
            p2.text = data["subtitle"]
            p2.font.size = Pt(16)
            p2.font.color.rgb = RGBColor(180, 205, 235)

        # Content Box
        content_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.2))
        tf_content = content_box.text_frame
        tf_content.word_wrap = True

        for idx, bullet in enumerate(data["bullets"]):
            p_b = tf_content.paragraphs[0] if idx == 0 else tf_content.add_paragraph()
            p_b.text = "•  " + bullet
            p_b.font.size = Pt(20)
            p_b.font.color.rgb = RGBColor(40, 40, 40)
            p_b.space_after = Pt(14)

        # Notes
        slide.notes_slide.notes_text_frame.text = data["notes"]

    os.makedirs("docs", exist_ok=True)
    output_pptx = os.path.join("docs", "internship_presentation.pptx")
    prs.save(output_pptx)
    print(f"Successfully generated PowerPoint presentation: {os.path.abspath(output_pptx)}")

if __name__ == "__main__":
    create_deck()
