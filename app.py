import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2, os, av, requests
import numpy as np
from keras.models import load_model
import gdown
import streamlit.components.v1 as components

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="🚦 Road Anomaly Detection", layout="wide")

# ===================== SESSION INIT =====================
if "page" not in st.session_state:
    st.session_state.page = "welcome"
if "rule_index" not in st.session_state:
    st.session_state.rule_index = 0
if "alert" not in st.session_state:
    st.session_state.alert = False

# ===================== MODEL LOAD =====================
MODEL_FILE = "road_anomaly_model.h5"
FILE_ID = "1FiHUDZPL1MFyG1g06_jjM4MJV2tH9rpg"

@st.cache_resource
def load_road_model():
    if not os.path.exists(MODEL_FILE):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_FILE, quiet=False)
    return load_model(MODEL_FILE)

model = load_road_model()
class_names = ['Accident', 'Fight', 'Fire', 'Snatching']
CONF_THRESHOLD = 0.85

# ===================== RTC CONFIG =====================
RTC_CONFIG = RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]})

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

        if confidence >= CONF_THRESHOLD:
            label = f"⚠️ {class_names[class_id]} ({confidence*100:.1f}%)"
            color = (0, 0, 255)
            self.anomaly_count += 1
            st.session_state.alert = True
        else:
            label = f"{class_names[class_id]} ({confidence*100:.1f}%)"
            color = (0, 255, 0)
            st.session_state.alert = False

        cv2.putText(img, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, f"Anomalies Detected: {self.anomaly_count}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===================== WEATHER FUNCTION =====================
def get_weather():
    try:
        latitude, longitude = 13.0827, 80.2707  # Chennai
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
        data = requests.get(url, timeout=5).json()
        return data.get("current_weather", None)
    except:
        return None

# ===================== FAMILY EMERGENCY =====================
FAMILY_NUMBERS = ["+919876543210", "+919812345678"]
def trigger_emergency():
    st.warning("🚨 Emergency signal sent to family members!")
    for number in FAMILY_NUMBERS:
        st.info(f"📞 Calling: {number}")

# ===================== WELCOME PAGE =====================
if st.session_state.page == "welcome":
    st.markdown("<h1 style='text-align:center;'>🚗 Happy Journey</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Drive safe, arrive happy</p>", unsafe_allow_html=True)
    if st.button("➡️ Continue"):
        st.session_state.page = "safety"
        st.experimental_rerun()

# ===================== SAFETY TIPS =====================
elif st.session_state.page == "safety":
    rules = [
        "🌤️ Ensure you are well-rested before starting your journey.",
        "🕶️ If you feel sleepy, take a short break and relax.",
        "🚰 Keep yourself hydrated and comfortable while driving.",
        "📵 Avoid distractions and focus on the road.",
        "❤️ Safety matters more than reaching early. Drive calmly."
    ]
    st.markdown(f"<div class='card'><h3>{rules[st.session_state.rule_index]}</h3></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Previous") and st.session_state.rule_index > 0:
            st.session_state.rule_index -= 1
            st.experimental_rerun()
    with col2:
        if st.session_state.rule_index < len(rules)-1:
            if st.button("Next ➡️"):
                st.session_state.rule_index += 1
                st.experimental_rerun()
        else:
            if st.button("🚗 Start Journey"):
                st.session_state.page = "main"
                st.experimental_rerun()

# ===================== MAIN DASHBOARD =====================
elif st.session_state.page == "main":
    st.markdown("<h1 style='text-align:center;'>🚗 Smart Road Safety System</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2.5, 1.5, 1.5])

    # ----------------- LIVE CAMERA -----------------
    with col1:
        st.markdown("<div class='card'><h3>🎥 Live Camera</h3></div>", unsafe_allow_html=True)
        webrtc_streamer(
            key="road-anomaly",
            video_processor_factory=AnomalyProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    # ----------------- STATUS + EMERGENCY -----------------
    with col2:
        st.markdown("<div class='card'><h3>🚦 Status</h3></div>", unsafe_allow_html=True)
        if st.session_state.alert:
            st.markdown("<h2 style='color:red;'>🚨 ANOMALY DETECTED!</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color:green;'>✅ All Clear</h2>", unsafe_allow_html=True)

        if st.button("🚨 Emergency Button"):
            trigger_emergency()

    # ----------------- WEATHER + HOTELS -----------------
    with col3:
        st.markdown("<div class='card'><h3>🌦️ Live Weather</h3></div>", unsafe_allow_html=True)
        weather = get_weather()
        if weather:
            st.write(f"🌡️ Temp: {weather['temperature']} °C")
            st.write(f"💨 Wind Speed: {weather['windspeed']} km/h")
        else:
            st.warning("Weather unavailable")

        st.markdown("<div class='card'><h3>🏨 Hotels Near Me</h3></div>", unsafe_allow_html=True)
        components.html(f"""
        <script>
        navigator.geolocation.getCurrentPosition(p=>{{
            document.getElementById("hotelmap").src=
            `https://maps.google.com/maps?q=hotels+near+${{p.coords.latitude}},${{p.coords.longitude}}&z=14&output=embed`;
        }});
        </script>
        <iframe id="hotelmap" width="100%" height="220" style="border-radius:12px;border:0;"></iframe>
        """, height=230)
