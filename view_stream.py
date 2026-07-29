import cv2
import os

def main():
    # Force OpenCV FFmpeg backend to use TCP for RTSP. 
    # This tunnels the video through the open 8554 port and bypasses UDP firewall blocks!
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    
    print("Waiting for direct UDP stream on port 5006...")
    # Listen directly on port 5006 for the UDP stream coming from the board!
    cap = cv2.VideoCapture("udp://@0.0.0.0:5006", cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("Error: Could not open the stream. Make sure the board is running!")
        return

    print("Stream connected! Press 'q' to close the window.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or dropped.")
            break
            
        cv2.imshow("Live AI Stream", frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
