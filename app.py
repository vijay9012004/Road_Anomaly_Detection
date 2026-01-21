import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import os
import av
import numpy as np
from keras.models import load_model
import gdown

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Road Anomaly Detection", layout="wide")
st.title("Real-Time Road Anomaly Detection (CNN)")

# ===================== CONSTANTS =====================
MODEL_FILE = "road_anomaly_model.h5"
FILE_ID = "1FiHUDZPL1MFyG1g06_jjM4MJV2tH9rpg"

CLASS_NAMES = ['Accident', 'Fight', 'Fire', 'Snatching']
CONF_THRESHOLD = 0.85

# ===================== MODEL DOWNLOAD =====================
def download_model():
    if not os.path.exists(MODEL_FILE):
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            MODEL_FILE,
            quiet=False
        )

download_model()

# ===================== RTC CONFIG =====================
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ===================== VIDEO PROCESSOR =====================
class AnomalyProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model(MODEL_FILE)
        self.anomaly_count = 0
        self.prev_anomaly = False

    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")

        # Preprocess
        img_input = cv2.resize(img, (224, 224))
        img_input = img_input.astype("float32") / 255.0
        img_input = np.expand_dims(img_input, axis=0)

        # Predict
        preds = self.model.predict(img_input, verbose=0)
        class_id = int(np.argmax(preds))
        confidence = float(np.max(preds))

        is_anomaly = confidence >= CONF_THRESHOLD

        # Count anomaly only once per event
        if is_anomaly and not self.prev_anomaly:
            self.anomaly_count += 1

        self.prev_anomaly = is_anomaly

        # Draw label
        label = f"{CLASS_NAMES[class_id]} ({confidence*100:.1f}%)"
        color = (0, 0, 255) if is_anomaly else (0, 255, 0)

        cv2.putText(img, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(img, f"Anomalies Detected: {self.anomaly_count}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===================== STREAM =====================
webrtc_streamer(
    key="road-anomaly",
    video_processor_factory=AnomalyProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
