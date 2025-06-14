
import streamlit as st
from deepface import DeepFace
import cv2
import os
import numpy as np
import pandas as pd
from tempfile import NamedTemporaryFile

st.set_page_config(
    page_title="FaceTimeMed - מעקב רגשי חכם",
    page_icon="📷",
    layout="centered"
)

st.markdown("""
<style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #4B8BBE;
        text-align: center;
        margin-bottom: 30px;
    }
    .sub {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">FaceTimeMed</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">מערכת יומית לניטור רגשי – פשוט לצלם ולהרגיש</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 העלאת סרטון יומי (mp4)", type=["mp4"])

if uploaded_file is not None:
    with st.spinner("🎬 מעבד את הסרטון... אנא המתן/י"):
        tfile = NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = fps
        frame_id = 0
        saved_frames = []
        os.makedirs("frames", exist_ok=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_interval == 0:
                frame_filename = f"frames/frame_{frame_id}.jpg"
                cv2.imwrite(frame_filename, frame)
                saved_frames.append(frame_filename)
                frame_id += 1
        cap.release()

        emotion_scores = []
        for frame_path in saved_frames:
            try:
                result = DeepFace.analyze(img_path=frame_path, actions=['emotion'], enforce_detection=False)
                emotion_scores.append(result[0]["emotion"])
            except:
                continue

    if emotion_scores:
        df = pd.DataFrame(emotion_scores)
        avg_emotions = df.mean().sort_values(ascending=False)
        st.success("🎯 ממוצע רגשי מהסרטון:")
        st.bar_chart(avg_emotions)
        st.markdown("""<div style='text-align:center; margin-top:20px; color:#888;'>
        רוצים לעקוב אחר מצבכם הרגשי לאורך זמן? חזרו לכאן מחר והעלו סרטון חדש 📆
        </div>""", unsafe_allow_html=True)
    else:
        st.error("❌ לא זוהו רגשות מהסרטון. מומלץ לנסות סרטון באור טוב ועם פנים ברורות.")
