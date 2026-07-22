import logging
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

logger = logging.getLogger(__name__)

def on_pad_caps_notify(pad, pspec, element_name):
    """Callback when pad caps change."""
    caps = pad.get_current_caps()
    pad_name = pad.get_name()
    logger.debug(f"=== [CAPS CHANGE] {element_name}:{pad_name} ===")
    if caps:
        logger.debug(caps.to_string())
    else:
        logger.debug("(no caps)")

def monitor_element_pads(element, element_name):
    """Monitor both src and sink pads of an element for caps changes."""
    sink_pad = element.get_static_pad("sink")
    if sink_pad:
        sink_pad.connect("notify::caps", on_pad_caps_notify, element_name)
    
    src_pad = element.get_static_pad("src")
    if src_pad:
        src_pad.connect("notify::caps", on_pad_caps_notify, element_name)

def print_key_element_caps(pipeline):
    """Print caps for all key elements we're interested in."""
    def print_pad_caps(element, pad_name, element_label, info=False):
        pad = element.get_static_pad(pad_name)
        if pad:
            caps = pad.get_current_caps()
            log = logger.info if info else logger.debug
            log(f"=== {element_label} {pad_name.upper()} CAPS ===")
            if caps:
                log(caps.to_string())
            else:
                log("(no caps yet)")
    
    it = pipeline.iterate_elements()
    while True:
        result, element = it.next()
        if result != Gst.IteratorResult.OK:
            break
        
        element_name = element.get_name()
        
        if "vpudec" in element_name:
            print_pad_caps(element, "src", "vpudec")
        elif "imxvideoconvert" in element_name:
            print_pad_caps(element, "sink", element_name)
            print_pad_caps(element, "src", element_name)
        elif "vpuenc" in element_name or element_name in ("file_enc", "stream_enc"):
            print_pad_caps(element, "sink", f"before vpuenc_h264 ({element_name})", info=True)
        elif element_name == "ml_sink":
            print_pad_caps(element, "sink", "appsink (ml_sink)")
        elif element_name == "overlay":
            print_pad_caps(element, "sink", "cairooverlay")
            print_pad_caps(element, "src", "after cairooverlay", info=True)

def handle_bus_message(bus, message, pipeline, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        logger.info("End of stream")
        pipeline.set_state(Gst.State.NULL)
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        logger.error(f"{err}: {debug}")
        pipeline.set_state(Gst.State.NULL)
        loop.quit()
    elif t == Gst.MessageType.STATE_CHANGED:
        old, new, pending = message.parse_state_changed()
        if message.src == pipeline and new == Gst.State.PLAYING:
            logger.info("=== [PIPELINE PLAYING] Printing current caps for key elements ===")
            print_key_element_caps(pipeline)

def build_pipeline(video_src: str, is_camera: bool, output_file: str, 
                   display_w: int, display_h: int, 
                   draw_cb, sample_cb, bus_cb, has_cairo: bool):
    from configs.settings import GST_BITRATE_FILE, GST_BITRATE_STREAM, GST_UDP_PORT
    
    if is_camera:
        source = f"v4l2src device={video_src} ! video/x-raw,framerate=30/1"
    else:
        source = f"filesrc location={video_src} ! qtdemux ! h264parse ! vpudec ! imxvideoconvert_g2d ! video/x-raw,format=BGRx"
        
    sink_str = "fakesink sync=false"
    
    if output_file:
        pipeline_str = (
            f"{source} ! "
            f"imxvideoconvert_g2d ! video/x-raw,width={display_w},height={display_h},format=BGRx ! "
            f"tee name=t "
            f"t. ! queue max-size-buffers=2 leaky=downstream ! cairooverlay name=overlay ! "
            f"tee name=out "
            f"out. ! queue max-size-buffers=10 leaky=downstream ! "
            f"imxvideoconvert_g2d ! videoconvert ! video/x-raw,format=I420 ! "
            f"vpuenc_h264 name=file_enc bitrate={GST_BITRATE_FILE} ! h264parse ! mp4mux ! filesink location={output_file} "
            f"out. ! queue max-size-buffers=2 leaky=downstream ! {sink_str} "
            f"out. ! queue max-size-buffers=2 leaky=downstream ! "
            f"imxvideoconvert_g2d ! videoconvert ! video/x-raw,format=I420 ! "
            f"vpuenc_h264 name=stream_enc bitrate={GST_BITRATE_STREAM} ! h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port={GST_UDP_PORT} "
            f"t. ! queue max-size-buffers=2 leaky=downstream ! "
            f"videoconvert ! video/x-raw,format=RGB ! appsink name=ml_sink emit-signals=true drop=true max-buffers=2 sync=false"
        )
    else:
        pipeline_str = (
            f"{source} ! "
            f"imxvideoconvert_g2d ! video/x-raw,width={display_w},height={display_h},format=BGRx ! "
            f"tee name=t "
            f"t. ! queue max-size-buffers=2 leaky=downstream ! cairooverlay name=overlay ! "
            f"tee name=out "
            f"out. ! queue max-size-buffers=2 leaky=downstream ! {sink_str} "
            f"out. ! queue max-size-buffers=2 leaky=downstream ! "
            f"imxvideoconvert_g2d ! videoconvert ! video/x-raw,format=I420 ! "
            f"vpuenc_h264 name=stream_enc bitrate={GST_BITRATE_STREAM} ! h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port={GST_UDP_PORT} "
            f"t. ! queue max-size-buffers=2 leaky=downstream ! "
            f"videoconvert ! video/x-raw,format=RGB ! appsink name=ml_sink emit-signals=true drop=true max-buffers=2 sync=false"
        )
        
    logger.info(f"Pipeline: {pipeline_str}")
    pipeline = Gst.parse_launch(pipeline_str)
    
    overlay = pipeline.get_by_name("overlay")
    if overlay and has_cairo:
        overlay.connect("draw", draw_cb)
    elif overlay:
        logger.warning("Python cairo not available; Cairo overlay callback disabled")
        
    ml_sink = pipeline.get_by_name("ml_sink")
    if ml_sink:
        ml_sink.connect("new-sample", sample_cb)
        
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_cb)
    
    return pipeline


def start_rtsp_server(udp_port, rtsp_port):
    """
    Start a GStreamer RTSP server that bridges the UDP stream from the
    main pipeline into an RTSP mount point at /video.

    Args:
        udp_port: The UDP port the main pipeline sends H.264 RTP to.
        rtsp_port: The port the RTSP server listens on.

    Returns:
        The GstRtspServer.RTSPServer instance (kept alive by caller).
    """
    from gi.repository import GstRtspServer

    server = GstRtspServer.RTSPServer()
    server.set_service(rtsp_port)

    factory = GstRtspServer.RTSPMediaFactory()
    factory.set_shared(True)
    # Bridge the UDP stream from the main pipeline into the RTSP server
    factory.set_launch(
        f'( udpsrc port={udp_port} caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96" ! '
        'rtph264pay name=pay0 pt=96 )'
    )

    mounts = server.get_mount_points()
    mounts.add_factory("/video", factory)
    server.attach(None)
    logger.info(f"[RTSP] Server running at rtsp://0.0.0.0:{rtsp_port}/video")

    return server
