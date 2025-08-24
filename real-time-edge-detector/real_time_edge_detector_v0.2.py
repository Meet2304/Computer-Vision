import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="Real-time Edge Detection", layout="wide")

# Sidebar configuration
st.sidebar.title("Settings")

# Filter selection
filter_type = st.sidebar.radio(
    "Choose Filter",
    ("Sobel", "Canny", "Laplacian of Gaussian")
)

# Parameters for filters
if filter_type == "Canny":
    threshold1 = st.sidebar.slider("Canny Threshold1", 50, 300, 100)
    threshold2 = st.sidebar.slider("Canny Threshold2", 50, 300, 200)
elif filter_type == "Sobel":
    ksize = st.sidebar.slider("Kernel Size", 1, 7, 3, step=2)
elif filter_type == "Laplacian of Gaussian":
    blur_ksize = st.sidebar.slider("Gaussian Blur Kernel", 1, 9, 3, step=2)
    lap_ksize = st.sidebar.slider("Laplacian Kernel Size", 1, 7, 3, step=2)


# --------------------------------
# Video Processor
# --------------------------------
class EdgeProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply chosen filter
        if filter_type == "Sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            sobel = cv2.magnitude(sobelx, sobely)
            sobel = np.uint8(np.clip(sobel, 0, 255))
            output = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

        elif filter_type == "Canny":
            edges = cv2.Canny(gray, threshold1, threshold2)
            output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        elif filter_type == "Laplacian of Gaussian":
            blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
            laplacian = cv2.Laplacian(blur, cv2.CV_64F, ksize=lap_ksize)
            laplacian = np.uint8(np.clip(np.absolute(laplacian), 0, 255))
            output = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)

        else:
            output = img

        # Overlay filter name
        cv2.putText(output, f"Filter: {filter_type}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(output, format="bgr24")


# --------------------------------
# Streamlit UI
# --------------------------------
st.title("📸 Real-time Edge Detection")
st.markdown(
    """
    This app performs **real-time edge detection** using:
    - Sobel Filter
    - Canny Edge Detector
    - Laplacian of Gaussian (LoG)

    ✅ Select **Filter** from the sidebar.  
    ✅ Adjust parameters live — changes take effect immediately.  
    """
)

webrtc_streamer(
    key="edge-detector",  # fixed key, no camera switching
    video_processor_factory=EdgeProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": []},  # no STUN
)
