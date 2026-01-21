import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2, os, av
import numpy as np
from keras.models import load_model
import gdown

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="🚦 Road Anomaly Detection", layout="wide")
st.title("🚨 Real-Time Road Anomaly Detection (CNN)")

# ===================== MODEL LOAD =====================
MODEL_FILE = "road_anomaly_model.h5"
FILE_ID = "1FiHUDZPL1MFyG1g06_jjM4MJV2tH9rpg"  # Google Drive file ID

@st.cache_resource
def load_road_model():
    """Load model, download if not exists"""
    if not os.path.exists(MODEL_FILE):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_FILE, quiet=False)
    return load_model(MODEL_FILE)

model = load_road_model()
class_names = ['Accident', 'Fight', 'Fire', 'Snatching']
CONF_THRESHOLD = 0.85

# ===================== RTC CONFIG =====================
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ===================== VIDEO PROCESSOR =====================
class AnomalyProcessor(VideoProcessorBase):
    def __init__(self):
        self.anomaly_count = 0

    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        img_input = cv2.resize(img, (224, 224)) / 255.0
        img_input = np.expand_dims(img_input, axis=0)

        pred = model.predict(img_input, verbose=0)
        class_id = np.argmax(pred)
        confidence = np.max(pred)

        # Draw prediction label
        if confidence >= CONF_THRESHOLD:
            label = f"⚠️ {class_names[class_id]} ({confidence*100:.1f}%)"
            color = (0, 0, 255)
            self.anomaly_count += 1
        else:
            label = f"{class_names[class_id]} ({confidence*100:.1f}%)"
            color = (0, 255, 0)

        cv2.putText(img, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, f"Anomalies Detected: {self.anomaly_count}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===================== STREAMLIT WEBCAM =====================
webrtc_streamer(
    key="road-anomaly",
    video_processor_factory=AnomalyProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
