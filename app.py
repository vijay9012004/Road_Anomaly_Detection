import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import os
import av
import numpy as np
from keras.models import load_model
import gdown
import tempfile

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Road Anomaly Detection", layout="wide")
st.title("Road Anomaly Detection using CNN")

# ===================== CONSTANTS =====================
MODEL_FILE = "road_anomaly_model.h5"
FILE_ID = "1FiHUDZPL1MFyG1g06_jjM4MJV2tH9rpg"
CLASS_NAMES = ['Accident', 'Fight', 'Fire', 'Snatching']
CONF_THRESHOLD = 0.85
IMG_SIZE = (224, 224)

# ===================== MODEL DOWNLOAD =====================
def download_model():
    if not os.path.exists(MODEL_FILE):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_FILE, quiet=False)

download_model()

@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_FILE)

model = load_cnn_model()

# ===================== PREPROCESS =====================
def preprocess_image(img):
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# ===================== PREDICTION FUNCTION =====================
def predict_anomaly(img):
    inp = preprocess_image(img)
    preds = model.predict(inp, verbose=0)
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))
    emergency = 1 if confidence >= CONF_THRESHOLD else 0
    return {"class": CLASS_NAMES[class_id], "confidence": confidence, "emergency": emergency}

# ===================== RTC CONFIG =====================
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# ===================== VIDEO PROCESSOR (LIVE WEBCAM) =====================
class AnomalyProcessor(VideoProcessorBase):
    def __init__(self):
        self.anomaly_count = 0
        self.prev_anomaly = False
        self.current_result = {"class": "", "confidence": 0, "emergency": 0}

    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        self.current_result = predict_anomaly(img)

        label = f"{self.current_result['class']} ({self.current_result['confidence']*100:.1f}%)"
        color = (0, 0, 255) if self.current_result["emergency"] else (0, 255, 0)

        if self.current_result["emergency"] and not self.prev_anomaly:
            self.anomaly_count += 1
        self.prev_anomaly = bool(self.current_result["emergency"])

        cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, f"Anomalies Detected: {self.anomaly_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(img, f"EMERGENCY: {self.current_result['emergency']}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===================== UI TABS =====================
tab1, tab2, tab3 = st.tabs(["Live Webcam", "Upload Video", "Upload Image"])

# ===================== TAB 1: LIVE WEBCAM =====================
with tab1:
    st.subheader("Live Road Anomaly Detection")
    processor = webrtc_streamer(
        key="road-anomaly",
        video_processor_factory=AnomalyProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,
    )

    if processor and processor.video_processor:
        result = processor.video_processor.current_result
        st.write("**Live Prediction Status**")
        st.write("Prediction:", result.get("class", ""))
        st.write("Confidence:", f"{result.get('confidence', 0)*100:.1f}%")
        if result.get("emergency", 0):
            st.error("🚨 EMERGENCY DETECTED")
        else:
            st.success("✅ Normal Condition")

# ===================== TAB 2: VIDEO UPLOAD =====================
with tab2:
    st.subheader("Video Upload Analysis")
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        if st.button("Predict Video"):
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            emergency_triggered = False

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                result = predict_anomaly(frame)
                label = f"{result['class']} ({result['confidence']*100:.1f}%)"
                color = (0, 0, 255) if result["emergency"] else (0, 255, 0)

                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"EMERGENCY: {result['emergency']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                stframe.image(frame, channels="BGR", use_container_width=True)

                if result["emergency"]:
                    emergency_triggered = True

            cap.release()

            if emergency_triggered:
                st.error("🚨 EMERGENCY DETECTED in Video")
            else:
                st.success("✅ No Critical Anomaly Detected in Video")

# ===================== TAB 3: IMAGE UPLOAD =====================
with tab3:
    st.subheader("Image Upload Analysis")
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if image_file is not None:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, channels="BGR", use_container_width=True)

        if st.button("Predict Image"):
            result = predict_anomaly(img)
            st.markdown(f"""
            **Prediction:** {result['class']}  
            **Confidence:** {result['confidence']*100:.2f}%  
            **Emergency Value:** {result['emergency']}
            """)
            if result["emergency"]:
                st.error("🚨 EMERGENCY DETECTED")
            else:
                st.success("✅ Normal Condition")
