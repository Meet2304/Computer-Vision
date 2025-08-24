import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="Real-time Edge Detection", layout="wide")

# Sidebar for selecting filter
st.sidebar.title("Edge Detection Settings")
filter_type = st.sidebar.radio(
    "Choose Filter",
    ("Sobel", "Canny", "Laplacian of Gaussian")
)

# Extra parameters
if filter_type == "Canny":
    threshold1 = st.sidebar.slider("Canny Threshold1", 50, 300, 100)
    threshold2 = st.sidebar.slider("Canny Threshold2", 50, 300, 200)
elif filter_type == "Sobel":
    ksize = st.sidebar.slider("Kernel Size", 1, 7, 3, step=2)
elif filter_type == "Laplacian of Gaussian":
    blur_ksize = st.sidebar.slider("Gaussian Blur Kernel", 1, 9, 3, step=2)
    lap_ksize = st.sidebar.slider("Laplacian Kernel Size", 1, 7, 3, step=2)


# Define Video Processor (new API)
class EdgeProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if filter_type == "Sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            sobel = cv2.magnitude(sobelx, sobely)
            sobel = np.uint8(np.clip(sobel, 0, 255))
            return av.VideoFrame.from_ndarray(
                cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR), format="bgr24"
            )

        elif filter_type == "Canny":
            edges = cv2.Canny(gray, threshold1, threshold2)
            return av.VideoFrame.from_ndarray(
                cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), format="bgr24"
            )

        elif filter_type == "Laplacian of Gaussian":
            blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
            laplacian = cv2.Laplacian(blur, cv2.CV_64F, ksize=lap_ksize)
            laplacian = np.uint8(np.clip(np.absolute(laplacian), 0, 255))
            return av.VideoFrame.from_ndarray(
                cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR), format="bgr24"
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.title("📸 Real-time Edge Detection")
st.markdown(
    """
    This app performs **real-time edge detection** using:
    - Sobel Filter
    - Canny Edge Detector
    - Laplacian of Gaussian (LoG)

    Use the sidebar to switch between filters and tune parameters.
    """
)

# Updated API call with STUN server config
webrtc_streamer(
    key="edge-detector",
    video_processor_factory=EdgeProcessor,
    media_stream_constraints={"video": True, "audio": False},  # local webcam only
    rtc_configuration={"iceServers": []},  # disable STUN → fixes aioice errors
)


